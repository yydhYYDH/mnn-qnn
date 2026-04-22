#!/usr/bin/env python3

import argparse
import math
import re
import struct
from pathlib import Path


RUN_DIR_RE = re.compile(r"^(graph(?:\d+|\d+_\d+))-run-(\d+)$")
LLAMA_META_RE = re.compile(r"^(graph(?:1_)?\d+)-(input|output)__.+\.meta\.txt$")
TENSOR_META_RE = re.compile(r"^(graph(?:1_)?\d+)-(input|output)__(t\d+)\.meta\.txt$")


def recurrent_input_attn_id(graph_index: int) -> int:
    return 129 + (graph_index - 1) * 93


def recurrent_input_hidden_id(graph_index: int) -> int:
    if graph_index == 1:
        return 4
    return 139 + (graph_index - 2) * 93


def recurrent_output_hidden_id(graph_index: int) -> int:
    return 139 + (graph_index - 1) * 93


def recurrent_output_q_id(graph_index: int) -> int:
    return 187 + (graph_index - 1) * 93


def recurrent_output_k_id(graph_index: int) -> int:
    return 216 + (graph_index - 1) * 93


def recurrent_output_v_id(graph_index: int) -> int:
    return 221 + (graph_index - 1) * 93


def recurrent_tensor_rules(graph_index: int) -> dict[str, str]:
    # Exported decode recurrent graphs follow a stable stride-93 naming rule:
    # graph1: in(t129,t4,t59,t90) out(t139,t187,t216,t221)
    # graph2: in(t222,t139,t59,t90) out(t232,t280,t309,t314)
    if graph_index < 1:
        raise ValueError(f"graph_index must be >= 1, got {graph_index}")

    return {
        "input_attn": f"t{recurrent_input_attn_id(graph_index)}",
        "input_hidden": f"t{recurrent_input_hidden_id(graph_index)}",
        "input_rope_a": "t59",
        "input_rope_b": "t90",
        "output_hidden": f"t{recurrent_output_hidden_id(graph_index)}",
        "output_q": f"t{recurrent_output_q_id(graph_index)}",
        "output_k": f"t{recurrent_output_k_id(graph_index)}",
        "output_v": f"t{recurrent_output_v_id(graph_index)}",
    }


def graph0_tensor_rules() -> dict[str, str]:
    return {
        "input_pos": "t0",
        "input_hidden": "t2",
        "output_aux0": "t4",
        "output_aux1": "t59",
        "output_aux2": "t90",
        "output_q": "t92",
        "output_k": "t122",
        "output_v": "t127",
    }


def final_block_tensor_rules() -> dict[str, str]:
    return {
        "input_attn": "t3384",
        "input_hidden": "t3301",
        "output_hidden": "t3393",
    }


def logits_graph_tensor_rules() -> dict[str, str]:
    return {
        "input_hidden": "t3394",
        "output_logits": "t3396",
    }


def read_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k.strip()] = v.strip()
    shape = [int(x) for x in meta.get("shape", "").split(",") if x]
    meta["shape"] = shape
    meta["bytes"] = int(meta.get("bytes", "0"))
    return meta


def read_values(path: Path, dtype: str, numel: int) -> list[float]:
    raw = path.read_bytes()
    if dtype == "float32":
        item_size = 4
        fmt = "<f"
    elif dtype == "float16":
        item_size = 2
        fmt = "<e"
    elif dtype == "int32":
        item_size = 4
        fmt = "<i"
    elif dtype == "uint32":
        item_size = 4
        fmt = "<I"
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    expected = numel * item_size
    if len(raw) != expected:
        raise ValueError(f"{path}: size mismatch, got {len(raw)} bytes, expected {expected} bytes")

    return [struct.unpack_from(fmt, raw, i)[0] for i in range(0, len(raw), item_size)]


def compare(lhs: list[float], rhs: list[float]) -> tuple[float, float, float]:
    max_abs = 0.0
    sum_sq = 0.0
    dot = 0.0
    lhs_sq = 0.0
    rhs_sq = 0.0

    for a, b in zip(lhs, rhs):
        diff = a - b
        abs_diff = abs(diff)
        max_abs = max(max_abs, abs_diff)
        sum_sq += diff * diff
        dot += a * b
        lhs_sq += a * a
        rhs_sq += b * b

    rmse = math.sqrt(sum_sq / len(lhs)) if lhs else 0.0
    cosine = dot / (math.sqrt(lhs_sq) * math.sqrt(rhs_sq)) if lhs_sq > 0 and rhs_sq > 0 else float("nan")
    sum_a, sum_b = sum(lhs), sum(rhs)
    return max_abs, rmse, cosine, sum_a, sum_b


def mnn_graph_to_llama_candidates(graph_name: str) -> list[str]:
    if graph_name == "graph1_0":
        return ["graph1_0", "graph0"]
    if not graph_name.startswith("graph"):
        return [graph_name]
    suffix = graph_name[5:]
    if "_" in suffix:
        return [graph_name]
    return [graph_name, f"graph1_{suffix}"]


def normalize_requested_graphs(requested_graphs: list[str] | None) -> tuple[list[str] | None, dict[str, str]]:
    if not requested_graphs:
        return None, {}

    normalized: list[str] = []
    display_name_by_graph: dict[str, str] = {}
    for graph_name in requested_graphs:
        if graph_name == "graph0":
            normalized_name = "graph1_0"
        elif graph_name.startswith("graph1_"):
            normalized_name = f"graph{graph_name[7:]}"
        else:
            normalized_name = graph_name
        normalized.append(normalized_name)
        display_name_by_graph[normalized_name] = graph_name

    return normalized, display_name_by_graph


def try_parse_graph_index(graph_name: str) -> int | None:
    if not graph_name.startswith("graph"):
        return None

    suffix = graph_name[5:]
    if not suffix:
        return None

    if suffix.isdigit():
        return int(suffix)

    if suffix.startswith("1_"):
        tail = suffix[2:]
        if tail.isdigit():
            return int(tail)

    return None


def semantic_name_for_tensor(graph_name: str, io_kind: str, tensor_name: str) -> str | None:
    rules: dict[str, str] = {}

    if graph_name in {"graph0", "graph1_0"}:
        rules = graph0_tensor_rules()
    else:
        graph_index = try_parse_graph_index(graph_name)
        if graph_index is None:
            return None
        if 1 <= graph_index <= 35:
            rules = recurrent_tensor_rules(graph_index)
        elif graph_index == 36:
            rules = final_block_tensor_rules()
        elif graph_index == 37:
            rules = logits_graph_tensor_rules()

    for semantic_name, rule_tensor_name in rules.items():
        if rule_tensor_name == tensor_name:
            if io_kind == "input" and semantic_name.startswith("input_"):
                return semantic_name
            if io_kind == "output" and semantic_name.startswith("output_"):
                return semantic_name
    return None


def format_tensor_with_semantic(meta_name: str) -> str:
    match = TENSOR_META_RE.match(meta_name)
    if not match:
        return meta_name

    graph_name, io_kind, tensor_name = match.groups()
    semantic_name = semantic_name_for_tensor(graph_name, io_kind, tensor_name)
    if semantic_name is None:
        return meta_name
    return f"{graph_name}-{io_kind}__{tensor_name}({semantic_name}).meta.txt"


def detect_graphs_and_run_id(mnn_root: Path, llama_dir: Path, requested_graphs: list[str] | None) -> tuple[list[str], int, dict[str, str]]:
    llama_graphs = set()
    for meta_path in llama_dir.glob("*.meta.txt"):
        m = LLAMA_META_RE.match(meta_path.name)
        if m:
            llama_graphs.add(m.group(1))

    normalized_requested, display_name_by_graph = normalize_requested_graphs(requested_graphs)

    run_ids_by_graph: dict[str, set[int]] = {}
    for path in mnn_root.iterdir():
        if not path.is_dir():
            continue
        m = RUN_DIR_RE.match(path.name)
        if not m:
            continue
        graph_name = m.group(1)
        run_id = int(m.group(2))
        run_ids_by_graph.setdefault(graph_name, set()).add(run_id)

    if normalized_requested:
        graphs = [g for g in normalized_requested if g in run_ids_by_graph]
    else:
        def graph_sort_key(name: str) -> tuple[int, int]:
            suffix = name[5:]
            if "_" in suffix:
                a, b = suffix.split("_", 1)
                return int(a), int(b)
            return int(suffix), -1
        graphs = sorted(run_ids_by_graph, key=graph_sort_key)

    llama_name_by_graph: dict[str, str] = {}
    filtered_graphs: list[str] = []
    for graph_name in graphs:
        matched_llama_name = None
        for candidate in mnn_graph_to_llama_candidates(graph_name):
            if candidate in llama_graphs:
                matched_llama_name = candidate
                break
        if matched_llama_name is None:
            continue
        filtered_graphs.append(graph_name)
        llama_name_by_graph[graph_name] = matched_llama_name

    graphs = filtered_graphs

    if not graphs:
        raise ValueError("no common graph dumps found in llama replay directory")

    common_run_ids = None
    for graph_name in graphs:
        if graph_name not in run_ids_by_graph:
            continue
        if common_run_ids is None:
            common_run_ids = set(run_ids_by_graph[graph_name])
        else:
            common_run_ids &= run_ids_by_graph[graph_name]

    if not common_run_ids:
        raise ValueError("failed to find a common MNN run id across selected graphs")

    return graphs, max(common_run_ids), llama_name_by_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MNN QNN decode dumps against llama.cpp replay dumps.")
    parser.add_argument("--mnn-root", required=True, help="root directory that contains graphN-run-XXXX subdirectories")
    parser.add_argument("--llama-dir", required=True, help="llama.cpp replay token directory")
    parser.add_argument("--graphs", nargs="*", help="optional subset such as: graph0 graph1 graph2")
    parser.add_argument("--run-id", type=int, help="explicit MNN run id; default auto-detect highest common run id")
    parser.add_argument("--atol", type=float, default=1e-3, help="max allowed absolute difference")
    parser.add_argument("--min-cosine", type=float, default=None, help="if set, cosine >= threshold is treated as pass")
    args = parser.parse_args()

    mnn_root = Path(args.mnn_root)
    llama_dir = Path(args.llama_dir)

    graphs, run_id, llama_name_by_graph = detect_graphs_and_run_id(mnn_root, llama_dir, args.graphs)
    if args.run_id is not None:
        run_id = args.run_id

    print(f"mnn_root:  {mnn_root}")
    print(f"llama_dir: {llama_dir}")
    print(f"run_id:    {run_id:04d}")
    print("graphs:")
    for graph_name in graphs:
        print(f"  {graph_name} <-> {llama_name_by_graph[graph_name]}")
    print("")

    recurrent_graphs = [g for g in graphs if (idx := try_parse_graph_index(g)) is not None and idx >= 1]
    if recurrent_graphs:
        print("recurrent tensor rules:")
        for graph_name in recurrent_graphs:
            graph_index = try_parse_graph_index(graph_name)
            assert graph_index is not None
            rules = recurrent_tensor_rules(graph_index)
            print(
                f"  {graph_name}: "
                f"in(attn={rules['input_attn']}, hidden={rules['input_hidden']}, "
                f"rope_a={rules['input_rope_a']}, rope_b={rules['input_rope_b']}) "
                f"out(hidden={rules['output_hidden']}, q={rules['output_q']}, "
                f"k={rules['output_k']}, v={rules['output_v']})"
            )
        print("")

    failures = []
    compared = 0
    cosine_pass = 0
    cosine_fail = 0

    for graph_name in graphs:
        mnn_graph_dir = mnn_root / f"{graph_name}-run-{run_id:04d}"
        if not mnn_graph_dir.exists():
            print(f"[skip] missing {mnn_graph_dir}")
            continue

        llama_graph_name = llama_name_by_graph[graph_name]
        for llama_meta in sorted(llama_dir.glob(f"{llama_graph_name}-*.meta.txt")):
            mnn_meta = mnn_graph_dir / llama_meta.name
            if not mnn_meta.exists():
                mnn_meta = mnn_graph_dir / llama_meta.name.replace(f"{llama_graph_name}-", f"{graph_name}-", 1)
            if not mnn_meta.exists():
                continue

            lhs_meta = read_meta(mnn_meta)
            rhs_meta = read_meta(llama_meta)
            tensor_display_name = format_tensor_with_semantic(llama_meta.name)

            if lhs_meta["dtype"] != rhs_meta["dtype"] or lhs_meta["shape"] != rhs_meta["shape"]:
                failures.append((llama_meta.name, float("inf"), "meta mismatch"))
                print(
                    f"[fail] {tensor_display_name} meta mismatch: "
                    f"mnn(shape={lhs_meta['shape']}, dtype={lhs_meta['dtype']}) "
                    f"llama(shape={rhs_meta['shape']}, dtype={rhs_meta['dtype']})"
                )
                continue

            numel = 1
            for dim in lhs_meta["shape"]:
                numel *= dim

            lhs = read_values(mnn_meta.with_suffix("").with_suffix(".bin"), lhs_meta["dtype"], numel)
            rhs = read_values(llama_meta.with_suffix("").with_suffix(".bin"), rhs_meta["dtype"], numel)
            max_abs, rmse, cosine, sum_a, sum_b = compare(lhs, rhs)
            compared += 1

            status = "pass" if max_abs <= args.atol else "fail"
            if args.min_cosine is not None and not math.isnan(cosine):
                status = "pass" if cosine >= args.min_cosine else "fail"
                if status == "pass":
                    cosine_pass += 1
                else:
                    cosine_fail += 1
            print(
                f"[{status}] {tensor_display_name} "
                f"shape={lhs_meta['shape']} dtype={lhs_meta['dtype']} "
                f"max_abs={max_abs:.3f} rmse={rmse:.3f} cosine={cosine:.3f} sum_a={sum_a:.4f} sum_b={sum_b:.4f}"
            )
            if status == "fail":
                failures.append((llama_meta.name, max_abs, "value mismatch"))

    print("")
    print(f"compared: {compared}")
    print(f"failures: {len(failures)}")
    if args.min_cosine is not None:
        print(f"min_cosine: {args.min_cosine}")
        print(f"cos_gte_threshold: {cosine_pass}")
        print(f"cos_lt_threshold: {cosine_fail}")

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
