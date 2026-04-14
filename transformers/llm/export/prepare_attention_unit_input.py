#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


GRAPH1_0_RULES = {
    "q": "t92",
    "k": "t122",
    "v": "t127",
}


def read_shape(meta_path: Path) -> list[int]:
    shape_text = None
    for line in meta_path.read_text().splitlines():
        if line.startswith("shape="):
            shape_text = line.split("=", 1)[1].strip()
            break
    if not shape_text:
        raise ValueError(f"missing shape= in {meta_path}")
    shape = [int(x) for x in shape_text.split(",") if x]
    if not shape:
        raise ValueError(f"empty shape in {meta_path}")
    return shape


def source_base(args: argparse.Namespace) -> tuple[Path, str]:
    if args.source_kind == "llama":
        return Path(args.source_root), args.graph
    run_dir = f"{args.graph}-run-{args.run_id:04d}"
    return Path(args.source_root) / run_dir, args.graph


def copy_tensor(src_dir: Path, src_graph: str, tensor_id: str, dst_dir: Path, dst_name: str) -> list[int]:
    src_bin = src_dir / f"{src_graph}-output__{tensor_id}.bin"
    src_meta = src_dir / f"{src_graph}-output__{tensor_id}.meta.txt"
    if not src_bin.is_file():
        raise FileNotFoundError(f"missing tensor bin: {src_bin}")
    if not src_meta.is_file():
        raise FileNotFoundError(f"missing tensor meta: {src_meta}")

    dst_bin = dst_dir / f"{dst_name}.bin"
    dst_meta = dst_dir / f"{dst_name}.meta.txt"
    shutil.copyfile(src_bin, dst_bin)
    shutil.copyfile(src_meta, dst_meta)
    return read_shape(src_meta)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract prefill q/k/v tensors for attention unit replay.")
    parser.add_argument("--source-kind", choices=["llama", "mnn"], default="llama")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--graph", default="graph1_0", help="prefill graph name or MNN run prefix, default: graph1_0")
    parser.add_argument("--run-id", type=int, default=0, help="MNN run id when --source-kind=mnn")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    src_dir, src_graph = source_base(args)
    dst_dir = Path(args.output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    q_shape = copy_tensor(src_dir, src_graph, GRAPH1_0_RULES["q"], dst_dir, "attn-query")
    k_shape = copy_tensor(src_dir, src_graph, GRAPH1_0_RULES["k"], dst_dir, "attn-key")
    v_shape = copy_tensor(src_dir, src_graph, GRAPH1_0_RULES["v"], dst_dir, "attn-value")

    if k_shape != v_shape:
        raise ValueError(f"k/v shape mismatch: {k_shape} vs {v_shape}")
    if len(q_shape) != 4 or len(k_shape) != 4:
        raise ValueError(f"expected rank-4 q/k tensors, got q={q_shape} k={k_shape}")
    if q_shape[0] != 1 or k_shape[0] != 1:
        raise ValueError(f"expected batch=1 tensors, got q={q_shape} k={k_shape}")
    if q_shape[3] != k_shape[3]:
        raise ValueError(f"head_dim mismatch: q={q_shape} k={k_shape}")

    env_path = dst_dir / "attention_input.env"
    env_path.write_text(
        "\n".join(
            [
                f"SEQ_LEN={q_shape[1]}",
                f"KV_LEN={k_shape[1]}",
                f"NUM_HEADS={q_shape[2]}",
                f"NUM_KV_HEADS={k_shape[2]}",
                f"HEAD_DIM={q_shape[3]}",
                "",
            ]
        )
    )

    print(f"prepared attention input at {dst_dir}")
    print(f"q_shape={q_shape}")
    print(f"k_shape={k_shape}")
    print(f"env_file={env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
