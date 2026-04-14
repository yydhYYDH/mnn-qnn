#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np


def parse_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def load_tensor(bin_path: Path, meta_path: Path) -> tuple[np.ndarray, dict[str, str]]:
    meta = parse_meta(meta_path)
    dtype_name = meta["dtype"].lower()
    if dtype_name in ("float32", "f32"):
        dtype = np.float32
    elif dtype_name in ("float16", "f16"):
        dtype = np.float16
    else:
        raise ValueError(f"unsupported dtype {dtype_name} for {bin_path}")

    shape = parse_shape(meta["shape"])
    arr = np.fromfile(bin_path, dtype=dtype)
    expected = math.prod(shape)
    if arr.size != expected:
        raise ValueError(f"{bin_path}: expected {expected} values, got {arr.size}")
    return arr.reshape(shape), meta


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs = lhs.astype(np.float32, copy=False)
    rhs = rhs.astype(np.float32, copy=False)
    diff = lhs - rhs
    abs_diff = np.abs(diff)
    denom = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0e-12)
    rel_diff = abs_diff / denom

    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    flat_lhs = lhs.reshape(-1)
    flat_rhs = rhs.reshape(-1)
    flat_abs = abs_diff.reshape(-1)
    flat_rel = rel_diff.reshape(-1)

    dot = float(np.dot(flat_lhs, flat_rhs))
    lhs_norm = float(np.linalg.norm(flat_lhs))
    rhs_norm = float(np.linalg.norm(flat_rhs))
    cosine = dot / (lhs_norm * rhs_norm) if lhs_norm > 0 and rhs_norm > 0 else float("nan")

    return {
        "max_abs": float(flat_abs[max_abs_idx]),
        "max_abs_idx": max_abs_idx,
        "max_rel": float(flat_rel[max_rel_idx]),
        "max_rel_idx": max_rel_idx,
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": cosine,
    }


def coords_from_index(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    return np.unravel_index(index, shape)


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no match for {pattern} under {root}")
    return matches[0]


def find_optional_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def print_compare(label: str, lhs_name: str, rhs_name: str, lhs: np.ndarray, rhs: np.ndarray) -> None:
    stats = compare_arrays(lhs, rhs)
    shape = lhs.shape
    print(f"== {label} ==")
    print(f"lhs: {lhs_name}")
    print(f"rhs: {rhs_name}")
    print(f"shape: {list(shape)}")
    print(f"max_abs_diff: {stats['max_abs']:.9g} at {list(coords_from_index(stats['max_abs_idx'], shape))}")
    print(f"max_rel_diff: {stats['max_rel']:.9g} at {list(coords_from_index(stats['max_rel_idx'], shape))}")
    print(f"rmse: {stats['rmse']:.9g}")
    print(f"cosine: {stats['cosine']:.9g}")
    print("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare layer-0 CPU attention dumps against graph0/graph1 tensors.")
    parser.add_argument("--dump-root", required=True, help="staged dump root, e.g. /home/reck/llama.cpp-test/qnn-prefill-attn-debug")
    parser.add_argument("--token-dir", default="decode-token-0000", help="token directory under dump-root")
    parser.add_argument("--cpu-attn-dir", default="", help="override CPU attention dir under dump-root")
    parser.add_argument("--layer", type=int, default=0, help="attention layer index to inspect")
    args = parser.parse_args()

    dump_root = Path(args.dump_root)
    token_dir = dump_root / args.token_dir
    if not token_dir.is_dir():
        raise FileNotFoundError(f"missing token dir: {token_dir}")

    if args.cpu_attn_dir:
        cpu_dir = dump_root / args.cpu_attn_dir
    else:
        cpu_dir = find_optional_one(dump_root, f"cpu-attn-layer{args.layer}-run-*")
    if cpu_dir is not None and not cpu_dir.is_dir():
        raise FileNotFoundError(f"missing cpu attention dir: {cpu_dir}")

    t92, _ = load_tensor(token_dir / "graph1_0-output__t92.bin", token_dir / "graph1_0-output__t92.meta.txt")
    t122, _ = load_tensor(token_dir / "graph1_0-output__t122.bin", token_dir / "graph1_0-output__t122.meta.txt")
    t127, _ = load_tensor(token_dir / "graph1_0-output__t127.bin", token_dir / "graph1_0-output__t127.meta.txt")
    t129, _ = load_tensor(token_dir / "graph1_1-input__t129.bin", token_dir / "graph1_1-input__t129.meta.txt")

    print(f"dump_root: {dump_root}")
    print(f"token_dir: {token_dir}")
    print(f"cpu_attn_dir: {cpu_dir if cpu_dir is not None else '<missing>'}")
    print("")
    print("Transform summary:")
    print("  t92/t122/t127 are graph0 Q/K/V dumps in [B, T, H, D] layout.")
    print("  t129 is the graph1 input_attn buffer in [B, T, H*D] layout.")
    print("  QNN boundary __fattn__ dump is [D, H, T, 1]; transpose(3, 0, 1, 2).reshape(B, T, H*D) should match t129 exactly.")
    print("  QNN boundary kqv_out dump is the post-attention contiguous tensor, but its memory layout is not the same as input_attn.")
    print("")

    boundary_q_path = find_optional_one(token_dir, f"attn-layer{args.layer}-boundary-src0-src0-contig__*.bin")
    boundary_k_path = find_optional_one(token_dir, f"attn-layer{args.layer}-boundary-src0-src1-contig__*.bin")
    boundary_v_path = find_optional_one(token_dir, f"attn-layer{args.layer}-boundary-src0-src2-contig__*.bin")
    boundary_out_path = find_optional_one(token_dir, f"attn-layer{args.layer}-boundary-src0-contig__*.bin")
    boundary_kqv_out_path = find_optional_one(token_dir, f"attn-layer{args.layer}-boundary-contig__*.bin")

    if boundary_q_path is not None:
        boundary_q, _ = load_tensor(boundary_q_path, boundary_q_path.with_suffix(".meta.txt"))
        boundary_q_graph_layout = np.transpose(boundary_q, (3, 1, 2, 0))
        print_compare("t92 vs boundary-q", "graph1_0-output__t92", boundary_q_path.name, t92, boundary_q_graph_layout)

    if boundary_k_path is not None:
        boundary_k, _ = load_tensor(boundary_k_path, boundary_k_path.with_suffix(".meta.txt"))
        boundary_k_prefix = boundary_k[:, : t122.shape[1], :, :]
        boundary_k_graph_layout = np.transpose(boundary_k_prefix, (3, 1, 2, 0))
        print_compare("t122 vs boundary-k", "graph1_0-output__t122", boundary_k_path.name, t122, boundary_k_graph_layout)

    if boundary_v_path is not None:
        boundary_v, _ = load_tensor(boundary_v_path, boundary_v_path.with_suffix(".meta.txt"))
        boundary_v_prefix = boundary_v[:, : t127.shape[1], :, :]
        boundary_v_graph_layout = np.transpose(boundary_v_prefix, (3, 1, 2, 0))
        print_compare("t127 vs boundary-v", "graph1_0-output__t127", boundary_v_path.name, t127, boundary_v_graph_layout)

    if boundary_out_path is not None:
        boundary_out, _ = load_tensor(boundary_out_path, boundary_out_path.with_suffix(".meta.txt"))
        boundary_out_flat = np.transpose(boundary_out, (3, 0, 1, 2)).reshape(t129.shape)
        print_compare("boundary-fattn vs t129", boundary_out_path.name, "graph1_1-input__t129", boundary_out_flat, t129)

    if boundary_kqv_out_path is not None:
        boundary_kqv_out, _ = load_tensor(boundary_kqv_out_path, boundary_kqv_out_path.with_suffix(".meta.txt"))
        boundary_kqv_out_flat = np.transpose(boundary_kqv_out, (3, 1, 0, 2)).reshape(t129.shape)
        print_compare("boundary-kqv_out vs t129", boundary_kqv_out_path.name, "graph1_1-input__t129", boundary_kqv_out_flat, t129)

    if cpu_dir is not None:
        attn_q_path = find_optional_one(cpu_dir, "attn-q__*.bin")
        attn_k_path = find_optional_one(cpu_dir, "attn-k__*.bin")
        attn_v_path = find_optional_one(cpu_dir, "attn-v__*.bin")
        attn_out_path = find_optional_one(cpu_dir, "attn-out__*.bin")

        if attn_q_path is not None:
            attn_q, _ = load_tensor(attn_q_path, attn_q_path.with_suffix(".meta.txt"))
            attn_q_graph_layout = np.transpose(attn_q, (0, 2, 1, 3))
            print_compare("t92 vs cpu-attn-q", "graph1_0-output__t92", attn_q_path.name, t92, attn_q_graph_layout)

        if attn_k_path is not None:
            attn_k, _ = load_tensor(attn_k_path, attn_k_path.with_suffix(".meta.txt"))
            attn_k_graph_layout = np.transpose(attn_k, (0, 2, 1, 3))
            print_compare("t122 vs cpu-attn-k", "graph1_0-output__t122", attn_k_path.name, t122, attn_k_graph_layout)

        if attn_v_path is not None:
            attn_v, _ = load_tensor(attn_v_path, attn_v_path.with_suffix(".meta.txt"))
            attn_v_graph_layout = np.transpose(attn_v, (0, 2, 1, 3))
            print_compare("t127 vs cpu-attn-v", "graph1_0-output__t127", attn_v_path.name, t127, attn_v_graph_layout)

        if attn_out_path is not None:
            attn_out, _ = load_tensor(attn_out_path, attn_out_path.with_suffix(".meta.txt"))
            attn_out_flat = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], attn_out.shape[2] * attn_out.shape[3])
            print_compare("cpu-attn-out vs t129", attn_out_path.name, "graph1_1-input__t129", attn_out_flat, t129)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
