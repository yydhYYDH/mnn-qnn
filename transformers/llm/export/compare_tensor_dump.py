#!/usr/bin/env python3

import argparse
import math
import struct
from pathlib import Path


def parse_shape(text: str) -> list[int]:
    shape = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not shape:
        raise ValueError("shape must not be empty")
    numel = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError(f"invalid shape dim: {dim}")
        numel *= dim
    return shape


def flatten_index_to_coords(index: int, shape: list[int]) -> list[int]:
    coords = [0] * len(shape)
    for i in range(len(shape) - 1, -1, -1):
        coords[i] = index % shape[i]
        index //= shape[i]
    return coords


def read_values(path: Path, dtype: str, numel: int) -> list[float]:
    raw = path.read_bytes()

    if dtype == "f32":
        item_size = 4
        fmt = "<f"
    elif dtype == "f16":
        item_size = 2
        fmt = "<e"
    elif dtype == "i32":
        item_size = 4
        fmt = "<i"
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    expected = numel * item_size
    if len(raw) != expected:
        raise ValueError(
            f"{path}: size mismatch, got {len(raw)} bytes, expected {expected} bytes"
        )

    return [struct.unpack_from(fmt, raw, i)[0] for i in range(0, len(raw), item_size)]


def compare(lhs: list[float], rhs: list[float]) -> dict:
    assert len(lhs) == len(rhs)

    max_abs = -1.0
    max_abs_idx = 0
    max_rel = -1.0
    max_rel_idx = 0

    sum_sq = 0.0
    dot = 0.0
    lhs_sq = 0.0
    rhs_sq = 0.0

    diffs = []

    for i, (a, b) in enumerate(zip(lhs, rhs)):
        diff = a - b
        abs_diff = abs(diff)
        denom = max(abs(a), abs(b), 1e-12)
        rel_diff = abs_diff / denom

        if abs_diff > max_abs:
            max_abs = abs_diff
            max_abs_idx = i
        if rel_diff > max_rel:
            max_rel = rel_diff
            max_rel_idx = i

        sum_sq += diff * diff
        dot += a * b
        lhs_sq += a * a
        rhs_sq += b * b
        diffs.append((abs_diff, i, a, b))

    rmse = math.sqrt(sum_sq / len(lhs)) if lhs else 0.0
    denom = math.sqrt(lhs_sq) * math.sqrt(rhs_sq)
    cosine = dot / denom if denom > 0 else float("nan")

    diffs.sort(reverse=True)

    return {
        "max_abs": max_abs,
        "max_abs_idx": max_abs_idx,
        "max_rel": max_rel,
        "max_rel_idx": max_rel_idx,
        "rmse": rmse,
        "cosine": cosine,
        "top_diffs": diffs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two raw tensor dumps with explicit dtype and shape."
    )
    parser.add_argument("--lhs", required=True, help="left-hand raw tensor file")
    parser.add_argument("--rhs", required=True, help="right-hand raw tensor file")
    parser.add_argument(
        "--dtype",
        required=True,
        choices=["f32", "f16", "i32"],
        help="raw tensor dtype",
    )
    parser.add_argument(
        "--shape",
        required=True,
        help="tensor shape, for example: 1,1,32,128",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="print top-k absolute differences",
    )
    args = parser.parse_args()

    shape = parse_shape(args.shape)
    numel = 1
    for dim in shape:
        numel *= dim

    lhs_path = Path(args.lhs)
    rhs_path = Path(args.rhs)

    lhs = read_values(lhs_path, args.dtype, numel)
    rhs = read_values(rhs_path, args.dtype, numel)

    stats = compare(lhs, rhs)

    max_abs_coords = flatten_index_to_coords(stats["max_abs_idx"], shape)
    max_rel_coords = flatten_index_to_coords(stats["max_rel_idx"], shape)

    print(f"lhs:   {lhs_path}")
    print(f"rhs:   {rhs_path}")
    print(f"dtype: {args.dtype}")
    print(f"shape: {shape}")
    print(f"numel: {numel}")
    print(f"max_abs_diff: {stats['max_abs']:.9g} at index {stats['max_abs_idx']} coords {max_abs_coords}")
    print(f"max_rel_diff: {stats['max_rel']:.9g} at index {stats['max_rel_idx']} coords {max_rel_coords}")
    print(f"rmse:         {stats['rmse']:.9g}")
    print(f"cosine:       {stats['cosine']:.9g}")
    print("")
    print("top diffs:")

    for rank, (abs_diff, index, a, b) in enumerate(stats["top_diffs"][: args.topk], start=1):
        coords = flatten_index_to_coords(index, shape)
        rel = abs_diff / max(abs(a), abs(b), 1e-12)
        print(
            f"{rank:2d}. idx={index:8d} coords={coords} "
            f"lhs={a:.9g} rhs={b:.9g} abs={abs_diff:.9g} rel={rel:.9g}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
