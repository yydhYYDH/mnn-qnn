#!/usr/bin/env python3

import argparse
import math
import re
import struct
from pathlib import Path


GRAPH_NAME_RE = re.compile(r"^graph\d+(?:_\d+)?$")
RUN_DIR_RE = re.compile(r"^(graph\d+(?:_\d+)?)-run-(\d+)$")
LLAMA_META_RE = re.compile(r"^(graph\d+(?:_\d+)?)-(input|output|source)__.+\.meta\.txt$")


def graph_sort_key(name: str) -> tuple[int, int]:
    suffix = name[5:]
    if "_" in suffix:
        major, minor = suffix.split("_", 1)
        return (int(major), int(minor))
    return (int(suffix), -1)


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
    return max_abs, rmse, cosine


def detect_graphs_and_run_id(mnn_root: Path, llama_dir: Path, requested_graphs: list[str] | None) -> tuple[list[str], int]:
    llama_graphs = set()
    for meta_path in llama_dir.glob("*.meta.txt"):
        m = LLAMA_META_RE.match(meta_path.name)
        if m:
            llama_graphs.add(m.group(1))

    if requested_graphs:
        graphs = [g for g in requested_graphs if g in llama_graphs]
    else:
        graphs = sorted(llama_graphs, key=graph_sort_key)

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

    return graphs, max(common_run_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MNN QNN decode dumps against llama.cpp replay dumps.")
    parser.add_argument("--mnn-root", required=True, help="root directory that contains graphN-run-XXXX subdirectories")
    parser.add_argument("--llama-dir", required=True, help="llama.cpp replay token directory")
    parser.add_argument("--graphs", nargs="*", help="optional subset such as: graph0 graph1 graph1_1")
    parser.add_argument("--run-id", type=int, help="explicit MNN run id; default auto-detect highest common run id")
    parser.add_argument("--atol", type=float, default=1e-3, help="max allowed absolute difference")
    args = parser.parse_args()

    mnn_root = Path(args.mnn_root)
    llama_dir = Path(args.llama_dir)

    graphs, run_id = detect_graphs_and_run_id(mnn_root, llama_dir, args.graphs)
    if args.run_id is not None:
        run_id = args.run_id

    print(f"mnn_root:  {mnn_root}")
    print(f"llama_dir: {llama_dir}")
    print(f"run_id:    {run_id:04d}")
    print(f"graphs:    {', '.join(graphs)}")
    print("")

    failures = []
    compared = 0

    for graph_name in graphs:
        mnn_graph_dir = mnn_root / f"{graph_name}-run-{run_id:04d}"
        if not mnn_graph_dir.exists():
            print(f"[skip] missing {mnn_graph_dir}")
            continue

        for llama_meta in sorted(llama_dir.glob(f"{graph_name}-*.meta.txt")):
            mnn_meta = mnn_graph_dir / llama_meta.name
            if not mnn_meta.exists():
                continue

            lhs_meta = read_meta(mnn_meta)
            rhs_meta = read_meta(llama_meta)

            if lhs_meta["dtype"] != rhs_meta["dtype"] or lhs_meta["shape"] != rhs_meta["shape"]:
                failures.append((llama_meta.name, float("inf"), "meta mismatch"))
                print(f"[fail] {llama_meta.name} meta mismatch: mnn={lhs_meta} llama={rhs_meta}")
                continue

            numel = 1
            for dim in lhs_meta["shape"]:
                numel *= dim

            lhs = read_values(mnn_meta.with_suffix("").with_suffix(".bin"), lhs_meta["dtype"], numel)
            rhs = read_values(llama_meta.with_suffix("").with_suffix(".bin"), rhs_meta["dtype"], numel)
            max_abs, rmse, cosine = compare(lhs, rhs)
            compared += 1

            status = "pass" if max_abs <= args.atol else "fail"
            print(
                f"[{status}] {llama_meta.name} "
                f"max_abs={max_abs:.9g} rmse={rmse:.9g} cosine={cosine:.9g}"
            )
            if status == "fail":
                failures.append((llama_meta.name, max_abs, "value mismatch"))

    print("")
    print(f"compared: {compared}")
    print(f"failures: {len(failures)}")

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
