import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def quant_header_len(bits: int, shape_int32: bool) -> int:
    shape_bytes = 8 if shape_int32 else 4
    map_len = 257 if bits == 8 else (1 << bits) + 1
    return 1 + shape_bytes + map_len


def unpack_bits(data: bytes, bits: int, count: int) -> np.ndarray:
    if bits == 8:
        arr = np.frombuffer(data, dtype=np.uint8, count=count)
        return arr.astype(np.int16)
    mask = (1 << bits) - 1
    out = np.empty(count, dtype=np.int16)
    bit_cursor = 0
    for idx in range(count):
        byte_index = bit_cursor // 8
        bit_in_byte = bit_cursor % 8
        shift = 8 - bits - bit_in_byte
        if shift >= 0:
            value = (data[byte_index] >> shift) & mask
        else:
            hi = (data[byte_index] << (-shift)) & 0xFF
            lo = data[byte_index + 1] >> (8 + shift)
            value = (hi | lo) & mask
        out[idx] = value
        bit_cursor += bits
    return out


def quant_value_map(bits: int) -> np.ndarray:
    offset = 1 << (bits - 1)
    return np.arange(-offset, offset, dtype=np.int16)


def fmt_number(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return str(value)
    return f"{value:.8g}"


def summarize_diff(lhs: np.ndarray, rhs: np.ndarray) -> Dict[str, float]:
    if lhs.shape != rhs.shape:
        return {
            "numel": int(lhs.size),
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "nonzero": int(max(lhs.size, rhs.size)),
            "equal": False,
            "cosine": float("nan"),
            "rel_diff": float("nan"),
        }
    lhs64 = lhs.astype(np.float64, copy=False).reshape(-1)
    rhs64 = rhs.astype(np.float64, copy=False).reshape(-1)
    diff = lhs64 - rhs64
    abs_diff = np.abs(diff)
    lhs_norm = np.linalg.norm(lhs64)
    rhs_norm = np.linalg.norm(rhs64)
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        cosine = 1.0 if lhs_norm == rhs_norm == 0.0 else 0.0
    else:
        cosine = float(np.dot(lhs64, rhs64) / (lhs_norm * rhs_norm))
    return {
        "numel": int(lhs.size),
        "max_abs": float(abs_diff.max(initial=0.0)),
        "mean_abs": float(abs_diff.mean() if abs_diff.size else 0.0),
        "nonzero": int(np.count_nonzero(abs_diff)),
        "equal": bool(np.array_equal(lhs, rhs)),
        "cosine": cosine,
        "rel_diff": float(diff.mean() if diff.size else 0.0) / (lhs64.mean()+1e-8),
    }


def preview_values(arr: np.ndarray, limit: int) -> str:
    flat = arr.reshape(-1)[:limit]
    return ", ".join(fmt_number(float(x)) for x in flat)


@dataclass
class TensorSegment:
    kind: str
    name: str
    values: np.ndarray


@dataclass
class OpRecord:
    name: str
    op_type: str
    meta: Dict[str, object]
    segments: List[TensorSegment]


def iter_weight_ops(graph: Dict[str, object]) -> Iterable[Dict[str, object]]:
    for op in graph["oplists"]:
        main = op.get("main", {})
        if op["type"] == "Convolution" and "external" in main:
            yield op
        elif op["type"] == "LayerNorm" and "external" in main:
            yield op
        elif op["type"] == "Const" and "external" in main:
            yield op


def read_layernorm_or_const(weight_file, op: Dict[str, object]) -> OpRecord:
    name = op["name"]
    op_type = op["type"]
    external = op["main"]["external"]
    weight_file.seek(external[0])
    segments = []
    if op_type == "LayerNorm":
        gamma = np.frombuffer(weight_file.read(external[1]), dtype=np.float32).copy()
        beta = np.frombuffer(weight_file.read(external[2]), dtype=np.float32).copy()
        segments.append(TensorSegment("gamma", name, gamma))
        segments.append(TensorSegment("beta", name, beta))
    else:
        values = np.frombuffer(weight_file.read(external[1]), dtype=np.float32).copy()
        segments.append(TensorSegment("float32", name, values))
    print(f"loaded {name} with {len(segments)} segments")
    return OpRecord(name=name, op_type=op_type, meta={"external": external}, segments=segments)


def read_convolution(weight_file, op: Dict[str, object], decode_dequant: bool) -> OpRecord:
    name = op["name"]
    main = op["main"]
    common = main["common"]
    external = main["external"]
    oc = int(common["outputCount"])
    ic = int(common["inputCount"])
    weight_len, alpha_len, bias_len = map(int, external[1:4])
    quant = main.get("quanParameter")
    weight_file.seek(external[0])
    segments: List[TensorSegment] = []
    meta: Dict[str, object] = {"external": external, "ic": ic, "oc": oc}

    if quant is None or int(quant.get("aMaxOrBits", 16)) == 16:
        weight = np.frombuffer(weight_file.read(weight_len), dtype=np.float16).copy().astype(np.float32)
        segments.append(TensorSegment("weight_fp16", name, weight))
        if alpha_len:
            alpha = np.frombuffer(weight_file.read(alpha_len), dtype=np.float32).copy()
            segments.append(TensorSegment("scale_bias", name, alpha))
        if bias_len:
            bias = np.frombuffer(weight_file.read(bias_len), dtype=np.float32).copy()
            segments.append(TensorSegment("bias", name, bias))
        return OpRecord(name=name, op_type="Convolution", meta=meta, segments=segments)

    bits = int(quant["aMaxOrBits"])
    shape_int32 = bool(quant.get("shapeInt32", False))
    header_len = quant_header_len(bits, shape_int32)
    header = weight_file.read(header_len)
    packed = weight_file.read(weight_len - header_len)
    blocks = alpha_len // (oc * 4)
    values_per_oc = ic
    q_count = oc * values_per_oc
    q_raw = unpack_bits(packed, bits, q_count).reshape(oc, values_per_oc)
    q_signed = quant_value_map(bits)[q_raw]
    scale = np.frombuffer(weight_file.read(alpha_len), dtype=np.float32).copy()
    bias = None
    if bias_len:
        bias = np.frombuffer(weight_file.read(bias_len), dtype=np.float32).copy()

    segments.append(TensorSegment("header_bytes", name, np.frombuffer(header, dtype=np.uint8).copy()))
    segments.append(TensorSegment("q_packed", name, np.frombuffer(packed, dtype=np.uint8).copy()))
    segments.append(TensorSegment("q_signed", name, q_signed))
    segments.append(TensorSegment("scale_bias", name, scale))
    if bias is not None:
        segments.append(TensorSegment("bias", name, bias))

    meta.update(
        {
            "bits": bits,
            "shape_int32": shape_int32,
            "header_len": header_len,
            "blocks": blocks,
            "readType": int(quant.get("readType", 0)),
            "aMin": int(quant.get("aMin", 0)),
        }
    )

    if decode_dequant and blocks > 0 and ic % blocks == 0:
        block_size = ic // blocks
        q_blocks = q_signed.reshape(oc, blocks, block_size).astype(np.float32)
        if scale.size == oc * blocks:
            dequant = q_blocks * scale.reshape(oc, blocks, 1)
            segments.append(TensorSegment("weight_dequant", name, dequant.reshape(oc, ic)))
            meta["dequant_mode"] = "symmetric"
        elif scale.size == oc * blocks * 2:
            scale_bias = scale.reshape(oc, blocks, 2)
            dequant = q_blocks * scale_bias[:, :, 1:2] + scale_bias[:, :, 0:1]
            segments.append(TensorSegment("weight_dequant", name, dequant.reshape(oc, ic)))
            meta["dequant_mode"] = "affine"
        else:
            meta["dequant_mode"] = "unknown"
    print(f"loaded {name} with {len(segments)} segments")
    return OpRecord(name=name, op_type="Convolution", meta=meta, segments=segments)


def load_records(graph_path: str, weight_path: str, decode_dequant: bool, read_limit: Optional[int] = None) -> Dict[str, OpRecord]:
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    records: Dict[str, OpRecord] = {}
    with open(weight_path, "rb") as weight_file:
        for op in iter_weight_ops(graph):
            if read_limit is not None and len(records) >= read_limit:
                print(f"reached read_limit of {read_limit}, breaking")
                break
            if op["type"] == "Convolution":
                records[op["name"]] = read_convolution(weight_file, op, decode_dequant)
            else:
                records[op["name"]] = read_layernorm_or_const(weight_file, op)
    return records


def mnn_to_hf_candidates(op: OpRecord) -> List[Tuple[str, str]]:
    name = op.name
    candidates: List[Tuple[str, str]] = []

    if name.startswith("/layers.") and "/self_attn/" in name and name.endswith("/Linear"):
        parts = name.strip("/").split("/")
        layer_id = parts[0].split(".")[1]
        proj_name = parts[2]
        candidates.append((f"model.layers.{layer_id}.self_attn.{proj_name}.weight", "weight_dequant"))
        candidates.append((f"model.layers.{layer_id}.self_attn.{proj_name}.bias", "bias"))
    elif name.startswith("/layers.") and "/mlp/" in name and name.endswith("/Linear"):
        parts = name.strip("/").split("/")
        layer_id = parts[0].split(".")[1]
        if len(parts) >= 4 and parts[2] == "shared_expert":
            proj_name = parts[3]
            candidates.append((f"model.layers.{layer_id}.mlp.shared_expert.{proj_name}.weight", "weight_dequant"))
            candidates.append((f"model.layers.{layer_id}.mlp.shared_expert.{proj_name}.bias", "bias"))
        else:
            proj_name = parts[2]
            candidates.append((f"model.layers.{layer_id}.mlp.{proj_name}.weight", "weight_dequant"))
            candidates.append((f"model.layers.{layer_id}.mlp.{proj_name}.bias", "bias"))
    elif name.startswith("/expert/"):
        parts = name.strip("/").split("/")
        layer_id, expert_id = parts[1].split("_", 1)
        proj_name = parts[2]
        candidates.append((f"model.layers.{layer_id}.mlp.experts.{expert_id}.{proj_name}.weight", "weight_dequant"))
        candidates.append((f"model.layers.{layer_id}.mlp.experts.{expert_id}.{proj_name}.bias", "bias"))
    elif "input_layernorm" in name:
        layer_id = name.split("/blocks.")[1].split("/")[0]
        candidates.append((f"model.layers.{layer_id}.input_layernorm.weight", "gamma"))
    elif "post_attention_layernorm" in name:
        layer_id = name.split("/blocks.")[1].split("/")[0]
        candidates.append((f"model.layers.{layer_id}.post_attention_layernorm.weight", "gamma"))
    elif name.startswith("/lm/lm_head/Linear"):
        candidates.append(("lm_head.weight", "weight_dequant"))
        candidates.append(("model.embed_tokens.weight", "weight_dequant"))
        candidates.append(("lm_head.bias", "bias"))
    elif "output_norm" in name or name.endswith("/norm/Mul_1_output_0"):
        candidates.append(("model.norm.weight", "gamma"))

    return candidates


class HFTensorLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._safe_open = None
        self._torch = None
        self._safetensor_index: Dict[str, str] = {}
        self._state_dict = None
        self._init_sources()

    def _require_safetensors(self):
        if self._safe_open is None:
            try:
                from safetensors import safe_open
            except ImportError as exc:
                raise RuntimeError("Need `safetensors` installed to compare against HF safetensor weights") from exc
            self._safe_open = safe_open
        return self._safe_open

    def _require_torch(self):
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("Need `torch` installed to compare against HF PyTorch weights") from exc
            self._torch = torch
        return self._torch

    def _init_sources(self):
        if os.path.isdir(self.model_path):
            safetensor_files = [
                os.path.join(self.model_path, filename)
                for filename in sorted(os.listdir(self.model_path))
                if filename.endswith(".safetensors")
            ]
            if safetensor_files:
                safe_open = self._require_safetensors()
                for path in safetensor_files:
                    with safe_open(path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            self._safetensor_index[key] = path
                return

            for filename in ("pytorch_model.bin", "model.bin", "consolidated.00.pth"):
                candidate = os.path.join(self.model_path, filename)
                if os.path.exists(candidate):
                    self.model_path = candidate
                    return

        if os.path.isfile(self.model_path) and self.model_path.endswith(".safetensors"):
            safe_open = self._require_safetensors()
            with safe_open(self.model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self._safetensor_index[key] = self.model_path

    def _load_state_dict(self):
        if self._state_dict is not None:
            return
        torch = self._require_torch()
        state = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported PyTorch checkpoint format: {type(state)}")
        self._state_dict = state

    def has_tensor(self, name: str) -> bool:
        if name in self._safetensor_index:
            return True
        if os.path.isfile(self.model_path) and self._state_dict is None and not self.model_path.endswith(".safetensors"):
            self._load_state_dict()
        return self._state_dict is not None and name in self._state_dict

    def get_tensor(self, name: str) -> np.ndarray:
        if name in self._safetensor_index:
            safe_open = self._require_safetensors()
            with safe_open(self._safetensor_index[name], framework="pt", device="cpu") as f:
                tensor = f.get_tensor(name)
            return tensor.detach().float().cpu().numpy()

        self._load_state_dict()
        if name not in self._state_dict:
            raise KeyError(name)
        tensor = self._state_dict[name]
        if hasattr(tensor, "detach"):
            return tensor.detach().float().cpu().numpy()
        return np.asarray(tensor, dtype=np.float32)


def print_equal_segments(left: OpRecord, right: OpRecord, preview_limit: int):
    seg_map_rhs = {seg.kind: seg for seg in right.segments}
    for seg in left.segments:
        peer = seg_map_rhs.get(seg.kind)
        if peer is None:
            continue
        print(f"  {seg.kind} lhs[:{preview_limit}]: {preview_values(seg.values, preview_limit)}")
        print(f"  {seg.kind} rhs[:{preview_limit}]: {preview_values(peer.values, preview_limit)}")


def compare_records(
    lhs: Dict[str, OpRecord],
    rhs: Dict[str, OpRecord],
    preview_limit: int,
    print_limit: Optional[int],
    only_diff: bool,
    show_values: bool,
) -> int:
    names = sorted(set(lhs) | set(rhs))
    total_diffs = 0
    shown = 0
    for name in names:
        left = lhs.get(name)
        right = rhs.get(name)
        if left is None or right is None:
            total_diffs += 1
            if print_limit is None or shown < print_limit:
                print(f"[MISSING] {name}")
                print(f"  lhs: {'present' if left else 'absent'}")
                print(f"  rhs: {'present' if right else 'absent'}")
                shown += 1
            continue

        seg_map_rhs = {seg.kind: seg for seg in right.segments}
        findings: List[Tuple[str, Dict[str, float], TensorSegment, TensorSegment]] = []
        missing_kinds = []
        for seg in left.segments:
            peer = seg_map_rhs.get(seg.kind)
            if peer is None:
                missing_kinds.append(seg.kind)
                continue
            stats = summarize_diff(seg.values, peer.values)
            if not stats["equal"]:
                findings.append((seg.kind, stats, seg, peer))

        if missing_kinds:
            total_diffs += 1
            if print_limit is None or shown < print_limit:
                print(f"[SEGMENT MISMATCH] {name}")
                print(f"  missing on rhs: {', '.join(missing_kinds)}")
                shown += 1
            continue

        if not findings and only_diff and not show_values:
            continue
        if findings:
            total_diffs += 1
        if print_limit is not None and shown >= print_limit:
            continue

        status = "DIFF" if findings else "OK"
        print(f"[{status}] {name} ({left.op_type})")
        meta_summary = ", ".join(f"{k}={v}" for k, v in left.meta.items() if k != "external")
        if meta_summary:
            print(f"  meta: {meta_summary}")
        print(f"  external(lhs): {left.meta.get('external')}")
        print(f"  external(rhs): {right.meta.get('external')}")

        if findings:
            for kind, stats, seg_l, seg_r in findings:
                print(
                    "  "
                    + f"{kind}: numel={stats['numel']} nonzero_diff={stats['nonzero']} "
                    + f"cosine={fmt_number(stats['cosine'])} max_abs={fmt_number(stats['max_abs'])} "
                    + f"mean_abs={fmt_number(stats['mean_abs'])} rel_diff={fmt_number(stats['rel_diff'])} "
                )
                print(f"    lhs[:{preview_limit}]: {preview_values(seg_l.values, preview_limit)}")
                print(f"    rhs[:{preview_limit}]: {preview_values(seg_r.values, preview_limit)}")
        else:
            print("  all compared segments are equal")
            if show_values:
                print_equal_segments(left, right, preview_limit)
        shown += 1
    return total_diffs


def compare_with_hf(
    records: Dict[str, OpRecord],
    hf_loader: HFTensorLoader,
    preview_limit: int,
    print_limit: Optional[int],
    only_diff: bool,
    show_values: bool,
) -> int:
    total_diffs = 0
    shown = 0
    for name in sorted(records):
        op = records[name]
        match = None
        for hf_name, segment_kind in mnn_to_hf_candidates(op):
            for seg in op.segments:
                if seg.kind == segment_kind and hf_loader.has_tensor(hf_name):
                    match = (hf_name, seg)
                    break
            if match is not None:
                break
        if match is None:
            continue

        hf_name, seg = match
        hf_values = hf_loader.get_tensor(hf_name)
        if seg.values.shape != hf_values.shape:
            total_diffs += 1
            if print_limit is None or shown < print_limit:
                print(f"[HF SHAPE] {name}")
                print(f"  segment: {seg.kind}")
                print(f"  hf tensor: {hf_name}")
                print(f"  mnn shape: {tuple(seg.values.shape)}")
                print(f"  hf shape: {tuple(hf_values.shape)}")
                shown += 1
            continue

        stats = summarize_diff(seg.values, hf_values)
        if not stats["equal"]:
            total_diffs += 1
        if not stats["equal"] or show_values or not only_diff:
            if print_limit is not None and shown >= print_limit:
                continue
            status = "HF-DIFF" if not stats["equal"] else "HF-OK"
            print(f"[{status}] {name}")
            print(f"  segment: {seg.kind}")
            print(f"  hf tensor: {hf_name}")
            print(
                "  "
                + f"numel={stats['numel']} nonzero_diff={stats['nonzero']} "
                + f"cosine={fmt_number(stats['cosine'])} max_abs={fmt_number(stats['max_abs'])} "
                + f"mean_abs={fmt_number(stats['mean_abs'])} rel_diff={fmt_number(stats['rel_diff'])} "
            )
            print(f"  mnn[:{preview_limit}]: {preview_values(seg.values, preview_limit)}")
            print(f"  hf[:{preview_limit}]: {preview_values(hf_values, preview_limit)}")
            shown += 1
    return total_diffs


def resolve_graph_path(weight_path: str, graph_path: Optional[str]) -> str:
    if graph_path:
        return graph_path
    candidate = f"{os.path.splitext(weight_path)[0]}.json"
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Cannot infer graph json for {weight_path}, please pass --graph-a/--graph-b")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read MNN external weight files via llm.mnn.json metadata and compare numeric contents."
    )
    parser.add_argument("--weight-a", required=True, help="Path to the first llm.mnn.weight")
    parser.add_argument("--weight-b", required=True, help="Path to the second llm.mnn.weight")
    parser.add_argument("--graph-a", help="Path to the first llm.mnn.json")
    parser.add_argument("--graph-b", help="Path to the second llm.mnn.json")
    parser.add_argument("--hf-model-path", help="Optional HF model path (directory or checkpoint file)")
    parser.add_argument("--preview", type=int, default=8, help="How many values to preview per diff segment")
    parser.add_argument("--op-limit", type=int, default=20, help="How many ops to print; use 0 for all")
    parser.add_argument("--read-limit", type=int, default=10, help="How many ops to load; use 0 for all")
    parser.add_argument("--all", action="store_true", help="Print equal ops too")
    parser.add_argument("--show-values", action="store_true", help="Print preview values even when tensors are equal")
    parser.add_argument(
        "--no-dequant",
        action="store_true",
        help="Skip reconstructing dequantized weights for quantized convolutions",
    )
    args = parser.parse_args()

    graph_a = resolve_graph_path(args.weight_a, args.graph_a)
    graph_b = resolve_graph_path(args.weight_b, args.graph_b)
    decode_dequant = not args.no_dequant
    print_limit = None if args.op_limit == 0 else args.op_limit

    records_a = load_records(graph_a, args.weight_a, decode_dequant, read_limit=print_limit)
    records_b = load_records(graph_b, args.weight_b, decode_dequant, read_limit=print_limit)
    diffs = compare_records(
        records_a,
        records_b,
        preview_limit=args.preview,
        print_limit=print_limit,
        only_diff=not args.all,
        show_values=args.show_values,
    )
    print()
    print(f"Compared ops: lhs={len(records_a)} rhs={len(records_b)} differing_ops={diffs}")

    hf_diffs = 0
    if args.hf_model_path:
        print("Compared with record a")
        hf_loader = HFTensorLoader(args.hf_model_path)
        hf_diffs = compare_with_hf(
            records_a,
            hf_loader,
            preview_limit=args.preview,
            print_limit=print_limit,
            only_diff=not args.all,
            show_values=args.show_values,
        )
        print()
        print(f"HF compared ops: lhs={len(records_a)} differing_ops={hf_diffs}")

        print("Compared with record b")
        hf_loader = HFTensorLoader(args.hf_model_path)
        hf_diffs = compare_with_hf(
            records_b,
            hf_loader,
            preview_limit=args.preview,
            print_limit=print_limit,
            only_diff=not args.all,
            show_values=args.show_values,
        )
        print()
        print(f"HF compared ops: lhs={len(records_a)} differing_ops={hf_diffs}")

    return 1 if (diffs or hf_diffs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
