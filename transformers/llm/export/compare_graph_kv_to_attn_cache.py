#!/usr/bin/env python3

import argparse
import itertools
import re
from pathlib import Path

import numpy as np


def recurrent_output_k_id(graph_index: int) -> int:
    return 216 + (graph_index - 1) * 93


def recurrent_output_v_id(graph_index: int) -> int:
    return 221 + (graph_index - 1) * 93


def recurrent_output_q_id(graph_index: int) -> int:
    return 187 + (graph_index - 1) * 93


def graph_qkv_sources_for_layer(layer: int) -> tuple[str, int, int, int]:
    if layer < 0:
        raise ValueError(f"layer must be >= 0, got {layer}")
    if layer == 0:
        return "graph1_0", 92, 122, 127

    graph_index = layer
    return (
        f"graph1_{graph_index}",
        recurrent_output_q_id(graph_index),
        recurrent_output_k_id(graph_index),
        recurrent_output_v_id(graph_index),
    )


def read_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k.strip()] = v.strip()
    meta["shape"] = [int(x) for x in meta.get("shape", "").split(",") if x]
    meta["bytes"] = int(meta.get("bytes", "0"))
    return meta


def np_dtype(dtype: str):
    if dtype in {"float32", "f32"}:
        return np.float32
    if dtype in {"float16", "f16"}:
        return np.float16
    if dtype in {"int32", "i32"}:
        return np.int32
    if dtype in {"uint32", "u32"}:
        return np.uint32
    raise ValueError(f"unsupported dtype: {dtype}")


def load_tensor(meta_path: Path, order: str) -> tuple[np.ndarray, dict]:
    meta = read_meta(meta_path)
    bin_path = meta_path.with_suffix("").with_suffix(".bin")
    arr = np.fromfile(bin_path, dtype=np_dtype(meta["dtype"])).reshape(meta["shape"], order=order)
    return arr.astype(np.float32), meta


def compare(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float, float]:
    lhs64 = lhs.astype(np.float64, copy=False).reshape(-1)
    rhs64 = rhs.astype(np.float64, copy=False).reshape(-1)
    diff = lhs64 - rhs64
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = np.linalg.norm(lhs64) * np.linalg.norm(rhs64)
    cosine = float(np.dot(lhs64, rhs64) / denom) if denom > 0 else float("nan")
    return max_abs, rmse, cosine


def find_one_regex(root: Path, pattern: str) -> Path:
    regex = re.compile(pattern)
    matches = sorted(path for path in root.iterdir() if regex.fullmatch(path.name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one match for {pattern}, got {len(matches)}")
    return matches[0]


def find_optional_one_regex(root: Path, pattern: str) -> Path | None:
    regex = re.compile(pattern)
    matches = sorted(path for path in root.iterdir() if regex.fullmatch(path.name))
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise FileNotFoundError(f"expected at most one match for {pattern}, got {len(matches)}")
    return matches[0]


def find_first_existing_regex(root: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        match = find_optional_one_regex(root, pattern)
        if match is not None:
            return match
    joined = " | ".join(patterns)
    raise FileNotFoundError(f"expected a match for one of: {joined}")


def to_attn_cache_slice(attn_raw: np.ndarray, slot: int) -> np.ndarray:
    # Cache dumps appear in either [D, KV, H, 1] or [D, H, KV, 1].
    if attn_raw.ndim != 4 or attn_raw.shape[3] != 1:
        raise ValueError(f"unexpected cache tensor shape: {list(attn_raw.shape)}")
    if slot < 0:
        raise ValueError(f"slot must be >= 0, got {slot}")

    # Older dumps: [D, KV, H, 1]
    if attn_raw.shape[1] > attn_raw.shape[2]:
        return np.transpose(attn_raw[:, slot : slot + 1, :, 0], (1, 2, 0))[None, ...]

    # Newer dumps: [D, H, KV, 1]
    return np.transpose(attn_raw[:, :, slot : slot + 1, 0], (2, 1, 0))[None, ...]


def to_attn_cur_slice(attn_raw: np.ndarray, token: int, num_tokens: int) -> np.ndarray:
    # Kcur/Qcur dumps may be either [D, H, T, 1] or [D, T, H, 1].
    if attn_raw.shape[2] == num_tokens:
        return np.transpose(attn_raw[:, :, token : token + 1, 0], (2, 1, 0))[None, ...]
    if attn_raw.shape[1] == num_tokens:
        return np.transpose(attn_raw[:, token : token + 1, :, 0], (1, 2, 0))[None, ...]
    raise ValueError(
        f"cannot infer token axis for tensor with shape {list(attn_raw.shape)} and num_tokens={num_tokens}"
    )


def to_attn_vcur_slice(attn_raw: np.ndarray, token: int, num_heads: int) -> np.ndarray:
    # non-flash attention Vcur dump layout: [D*HKV, T, 1, 1]
    # Recover the flattened [D, HKV] view and transpose to [1, 1, HKV, D].
    flat = attn_raw[:, token, 0, 0]
    if flat.size % num_heads != 0:
        raise ValueError(f"Vcur flat size {flat.size} is not divisible by num_heads={num_heads}")
    head_dim = flat.size // num_heads
    return flat.reshape(head_dim, num_heads).T[None, None, ...]


def to_attn_q_slice(attn_raw: np.ndarray, token: int, num_tokens: int) -> np.ndarray:
    return to_attn_cur_slice(attn_raw, token, num_tokens)


def to_attn_q_graph(attn_raw: np.ndarray, num_tokens: int) -> np.ndarray:
    if attn_raw.shape[2] == num_tokens:
        return np.transpose(attn_raw, (3, 2, 1, 0))
    if attn_raw.shape[1] == num_tokens:
        return np.transpose(attn_raw, (3, 1, 2, 0))
    raise ValueError(
        f"cannot infer token axis for Q tensor with shape {list(attn_raw.shape)} and num_tokens={num_tokens}"
    )


def summarize_full_tensor_pair(name: str, lhs: np.ndarray, rhs: np.ndarray) -> None:
    max_abs, rmse, cosine = compare(lhs, rhs)
    print(f"{name}:")
    print(f"  lhs shape={list(lhs.shape)}")
    print(f"  rhs shape={list(rhs.shape)}")
    print(
        f"  max_abs={max_abs:.6f} rmse={rmse:.6f} cosine={cosine:.6f} "
        f"sum(lhs)={np.sum(lhs):.6f} sum(rhs)={np.sum(rhs):.6f}"
    )


def summarize_axis_permutations(name: str, lhs: np.ndarray, rhs_raw: np.ndarray, limit: int = 6) -> None:
    print(f"{name}:")
    print(f"  lhs shape={list(lhs.shape)}")
    print(f"  rhs raw shape={list(rhs_raw.shape)}")

    candidates = []
    for perm in itertools.permutations(range(rhs_raw.ndim)):
        rhs = np.transpose(rhs_raw, perm)
        if rhs.shape != lhs.shape:
            continue
        max_abs, rmse, cosine = compare(lhs, rhs)
        candidates.append((rmse, perm, max_abs, cosine, float(np.sum(rhs))))

    if not candidates:
        print("  no axis permutation produces the target shape")
        return

    candidates.sort(key=lambda item: item[0])
    for rmse, perm, max_abs, cosine, rhs_sum in candidates[:limit]:
        print(
            f"  perm={perm} max_abs={max_abs:.6f} rmse={rmse:.6f} "
            f"cosine={cosine:.6f} sum(rhs)={rhs_sum:.6f}"
        )


def load_q_tensor_best_order(meta_path: Path, graph_q: np.ndarray) -> tuple[np.ndarray, dict, str]:
    best_arr = None
    best_meta = None
    best_order = None
    best_rmse = None

    for order in ("C", "F"):
        arr, meta = load_tensor(meta_path, order=order)
        try:
            q_graph = to_attn_q_graph(arr, graph_q.shape[1])
        except ValueError:
            continue
        _, rmse, _ = compare(graph_q, q_graph)
        if best_rmse is None or rmse < best_rmse:
            best_arr = arr
            best_meta = meta
            best_order = order
            best_rmse = rmse

    if best_arr is None or best_meta is None or best_order is None:
        raise ValueError(f"failed to load Q tensor with a compatible layout from {meta_path}")

    return best_arr, best_meta, best_order


def summarize_q_pair(name: str, graph_tensor: np.ndarray, attn_tensor: np.ndarray, limit: int) -> None:
    print(f"{name}:")
    print(f"  graph shape={list(graph_tensor.shape)}")
    print(f"  attn  shape={list(attn_tensor.shape)}")

    token_count = min(graph_tensor.shape[1], limit if limit >= 0 else graph_tensor.shape[1])
    for token in range(token_count):
        graph_slice = graph_tensor[:, token : token + 1, :, :]
        attn_slice = to_attn_q_slice(attn_tensor, token, graph_tensor.shape[1])
        max_abs, rmse, cosine = compare(graph_slice, attn_slice)
        print(
            f"  token={token:3d} "
            f"max_abs={max_abs:.6f} rmse={rmse:.6f} cosine={cosine:.6f} sum(graph)={np.sum(graph_slice):.6f}"
            f" sum(attn)={np.sum(attn_slice):.6f}"
        )


def summarize_pair(name: str, graph_tensor: np.ndarray, attn_tensor: np.ndarray, slots: list[int], limit: int) -> None:
    print(f"{name}:")
    print(f"  graph shape={list(graph_tensor.shape)}")
    print(f"  attn  shape={list(attn_tensor.shape)}")

    token_count = min(graph_tensor.shape[1], limit if limit >= 0 else graph_tensor.shape[1], len(slots))
    for token in range(token_count):
        slot = slots[token]
        graph_slice = graph_tensor[:, token : token + 1, :, :]
        attn_slice = to_attn_cache_slice(attn_tensor, slot)
        max_abs, rmse, cosine = compare(graph_slice, attn_slice)
        print(
            f"  token={token:3d} slot={slot:3d} "
            f"max_abs={max_abs:.6f} rmse={rmse:.6f} cosine={cosine:.6f} sum(graph)={np.sum(graph_slice):.6f}"
            f" sum(attn)={np.sum(attn_slice):.6f}"
        )
        # print(attn_slice[0, 0, 0, :3])


def summarize_cur_pair(name: str, graph_tensor: np.ndarray, attn_tensor: np.ndarray, limit: int) -> None:
    print(f"{name}:")
    print(f"  graph shape={list(graph_tensor.shape)}")
    print(f"  attn  shape={list(attn_tensor.shape)}")

    token_count = min(graph_tensor.shape[1], limit if limit >= 0 else graph_tensor.shape[1])
    for token in range(token_count):
        graph_slice = graph_tensor[:, token : token + 1, :, :]
        attn_slice = to_attn_cur_slice(attn_tensor, token, graph_tensor.shape[1])
        max_abs, rmse, cosine = compare(graph_slice, attn_slice)
        print(
            f"  token={token:3d} "
            f"max_abs={max_abs:.6f} rmse={rmse:.6f} cosine={cosine:.6f} sum(graph)={np.sum(graph_slice):.6f}"
            f" sum(attn)={np.sum(attn_slice):.6f}"
        )


def summarize_vcur_pair(name: str, graph_tensor: np.ndarray, attn_tensor: np.ndarray, limit: int) -> None:
    print(f"{name}:")
    print(f"  graph shape={list(graph_tensor.shape)}")
    print(f"  attn  shape={list(attn_tensor.shape)}")

    token_count = min(graph_tensor.shape[1], attn_tensor.shape[1], limit if limit >= 0 else graph_tensor.shape[1])
    num_heads = graph_tensor.shape[2]
    for token in range(token_count):
        graph_slice = graph_tensor[:, token : token + 1, :, :]
        attn_slice = to_attn_vcur_slice(attn_tensor, token, num_heads)
        max_abs, rmse, cosine = compare(graph_slice, attn_slice)
        print(
            f"  token={token:3d} "
            f"max_abs={max_abs:.6f} rmse={rmse:.6f} cosine={cosine:.6f} sum(graph)={np.sum(graph_slice):.6f}"
            f" sum(attn)={np.sum(attn_slice):.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare graph output K/V tensors against the K/V tensors actually read by a given attention layer."
    )
    parser.add_argument("--llama-dir", required=True, help="dump dir such as /tmp/llama-prefill-compare/decode-token-0000")
    parser.add_argument("--layer", type=int, default=1, help="attention layer index, default 1")
    parser.add_argument(
        "--slots",
        default="0,1,2,3,4,5,6,7",
        help="comma-separated physical KV slots to compare against token 0..N-1",
    )
    parser.add_argument("--limit", type=int, default=8, help="how many token/slot pairs to print, default 8")
    args = parser.parse_args()

    root = Path(args.llama_dir)
    layer = args.layer
    slots = [int(x) for x in args.slots.split(",") if x.strip()]
    graph_name, graph_q_id, graph_k_id, graph_v_id = graph_qkv_sources_for_layer(layer)

    graph_q_path = root / f"{graph_name}-output__t{graph_q_id}.meta.txt"
    graph_k_path = root / f"{graph_name}-output__t{graph_k_id}.meta.txt"
    graph_v_path = root / f"{graph_name}-output__t{graph_v_id}.meta.txt"
    attn_q_path = find_optional_one_regex(
        root,
        rf"attn-layer{layer}-loghit-q-permuted-.*__Qcur-{layer}(?:__.*)?\.meta\.txt",
    )
    if attn_q_path is None:
        attn_q_path = find_optional_one_regex(
            root,
            rf"attn-layer{layer}-q-permuted__Qcur-{layer}(?:__.*)?\.meta\.txt",
        )
    if attn_q_path is None:
        attn_q_path = find_optional_one_regex(
            root,
            rf"attn-layer{layer}-src0-src0-src1-src0-src1__Qcur-{layer}(?:__.*)?\.meta\.txt",
        )
    if attn_q_path is None:
        attn_q_path = find_optional_one_regex(
            root,
            rf"attn-layer{layer}-loghit-q-.*__Qcur-{layer}(?:__.*)?\.meta\.txt",
        )
    if attn_q_path is None:
        attn_q_path = find_optional_one_regex(
            root,
            rf"attn-layer{layer}-boundary-src0-src0-contig__Qcur-{layer}(?:__.*)?\.meta\.txt",
        )
    if attn_q_path is None:
        attn_q_path = find_optional_one_regex(
            root,
            rf"attn-layer{layer}-src0-src0__Qcur-{layer}(?:__.*)?\.meta\.txt",
        )
    attn_k_path = find_first_existing_regex(
        root,
        [
            rf"attn-layer{layer}-cache-k__kv_cache_k-{layer}(?:__.*)?\.meta\.txt",
            rf"attn-layer{layer}-boundary-src0-src1-contig__kv_cache_k-{layer}(?:__.*)?\.meta\.txt",
            rf"attn-layer{layer}-src0-src1__kv_cache_k-{layer}(?:__.*)?\.meta\.txt",
            rf"attn-layer{layer}-src0-src1__Kcur-{layer}(?:__.*)?\.meta\.txt",
        ],
    )
    attn_v_path = find_first_existing_regex(
        root,
        [
            rf"attn-layer{layer}-cache-v__kv_cache_v-{layer}(?:__.*)?\.meta\.txt",
            rf"attn-layer{layer}-boundary-src0-src2-contig__kv_cache_v-{layer}(?:__.*)?\.meta\.txt",
            rf"attn-layer{layer}-src0-src2__kv_cache_v-{layer}(?:__.*)?\.meta\.txt",
            rf"attn-layer{layer}-src0-src2__Vcur-{layer}(?:__.*)?\.meta\.txt",
        ],
    )

    graph_q, _ = load_tensor(graph_q_path, order="C")
    graph_k, _ = load_tensor(graph_k_path, order="C")
    graph_v, _ = load_tensor(graph_v_path, order="C")
    graph0_q_after_path = find_optional_one_regex(
        root,
        rf"graph0-dst-q-after-copy(?:__.*)?\.meta\.txt",
    )
    q_base_path = find_optional_one_regex(
        root,
        rf"attn-layer{layer}-q-base__Qcur-{layer}(?:__.*)?\.meta\.txt",
    )
    q_view_path = find_optional_one_regex(
        root,
        rf"attn-layer{layer}-q-view__Qcur-{layer}(?:__.*)?\.meta\.txt",
    )
    q_perm_path = find_optional_one_regex(
        root,
        rf"attn-layer{layer}-q-permuted__Qcur-{layer}(?:__.*)?\.meta\.txt",
    )
    attn_k_order = "F" if "kv_cache_k-" in attn_k_path.name else "C"
    attn_v_order = "F" if "kv_cache_v-" in attn_v_path.name else "C"
    attn_k, _ = load_tensor(attn_k_path, order=attn_k_order)
    attn_v, _ = load_tensor(attn_v_path, order=attn_v_order)
    attn_q = None
    attn_q_order = None
    if attn_q_path is not None:
        attn_q, _, attn_q_order = load_q_tensor_best_order(attn_q_path, graph_q)
    graph0_q_after = None
    if graph0_q_after_path is not None:
        graph0_q_after, _ = load_tensor(graph0_q_after_path, order="C")
    q_base = None
    if q_base_path is not None:
        q_base, _ = load_tensor(q_base_path, order="C")
    q_view = None
    q_view_meta = None
    if q_view_path is not None:
        q_view, q_view_meta = load_tensor(q_view_path, order="C")
    q_perm = None
    if q_perm_path is not None:
        q_perm, _, _ = load_q_tensor_best_order(q_perm_path, graph_q)

    print(f"llama_dir={root}")
    print(f"layer={layer}")
    print(f"slots={slots}")
    print(f"graph_name={graph_name}")
    print(f"graph_q={graph_q_path.name}")
    print(f"graph_k={graph_k_path.name}")
    print(f"graph_v={graph_v_path.name}")
    print(f"attn_q={attn_q_path.name if attn_q_path is not None else '<missing>'}")
    if attn_q_order is not None:
        print(f"attn_q_order={attn_q_order}")
    print(f"attn_k={attn_k_path.name}")
    print(f"attn_v={attn_v_path.name}")
    print(f"graph0_q_after={graph0_q_after_path.name if graph0_q_after_path is not None else '<missing>'}")
    print(f"q_base={q_base_path.name if q_base_path is not None else '<missing>'}")
    print(f"q_view={q_view_path.name if q_view_path is not None else '<missing>'}")
    if q_view is not None and q_view_meta is not None:
        print(
            "q_view_meta:"
            f" dtype={q_view_meta.get('dtype', '<missing>')}"
            f" shape={q_view_meta.get('shape', [])}"
            f" bytes={q_view_meta.get('bytes', '<missing>')}"
        )
        print(f"q_view_numpy: dtype={q_view.dtype} shape={list(q_view.shape)}")
    print(f"q_perm={q_perm_path.name if q_perm_path is not None else '<missing>'}")
    print("")
    if attn_q is not None:
        print("sum of graph_q:", np.sum(graph_q))
        print("sum of attn_q:", np.sum(attn_q))
        summarize_q_pair(f"Q: {graph_name}-output__t{graph_q_id} vs attn-layer{layer}-src0-src0", graph_q, attn_q, args.limit)
        print("")
    else:
        print("Q: skipped, attn_q dump not found")
        print("")
    if graph0_q_after is not None:
        graph0_q_after_graph = np.transpose(graph0_q_after, (3, 2, 1, 0))
        summarize_full_tensor_pair(
            "Q full: graph_t92 vs graph0-dst-q-after-copy",
            graph_q[:, : graph0_q_after.shape[2], :, :],
            graph0_q_after_graph,
        )
        print("")
    if q_base is not None:
        summarize_full_tensor_pair(
            "Q full: graph_t92 vs q_base",
            graph_q[:, : q_base.shape[2], :, :],
            np.transpose(q_base, (3, 2, 1, 0)),
        )
        print("")
    if q_view is not None:
        summarize_full_tensor_pair(
            "Q full: graph_t92 vs q_view",
            graph_q[:, : q_view.shape[2], :, :],
            np.transpose(q_view, (3, 2, 1, 0)),
        )
        summarize_axis_permutations(
            "Q full: graph_t92 vs q_view axis search",
            graph_q,
            q_view,
        )
        print("")
    if q_perm is not None:
        summarize_full_tensor_pair(
            "Q full: graph_t92 vs q_permuted",
            graph_q[:, : q_perm.shape[1], :, :],
            to_attn_q_graph(q_perm, graph_q.shape[1])[:, : q_perm.shape[1], :, :],
        )
        print("")
    print("sum of graph_k:", np.sum(graph_k))
    print("sum of attn_k:", np.sum(attn_k))
    if "kv_cache_k-" in attn_k_path.name:
        summarize_pair(f"K: {graph_name}-output__t{graph_k_id} vs attn-layer{layer}-src0-src1", graph_k, attn_k, slots, args.limit)
    else:
        summarize_cur_pair(f"K: {graph_name}-output__t{graph_k_id} vs attn-layer{layer}-src0-src1", graph_k, attn_k, args.limit)
    print("")
    print("sum of graph_v:", np.sum(graph_v))
    print("sum of attn_v:", np.sum(attn_v))
    if "kv_cache_v-" in attn_v_path.name:
        summarize_pair(f"V: {graph_name}-output__t{graph_v_id} vs attn-layer{layer}-src0-src2", graph_v, attn_v, slots, args.limit)
    else:
        summarize_vcur_pair(f"V: {graph_name}-output__t{graph_v_id} vs attn-layer{layer}-src0-src2", graph_v, attn_v, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
