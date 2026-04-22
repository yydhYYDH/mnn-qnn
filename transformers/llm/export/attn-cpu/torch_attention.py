import numpy as np
import torch
import os
import argparse
from pathlib import Path

def load_tensor_from_bin(path, shape):
    """Load raw float32 .bin file into torch tensor with given shape."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    if data.size != np.prod(shape):
        raise ValueError(f"Expected {np.prod(shape)} elements, got {data.size} in {path}")
    return torch.from_numpy(data).view(shape)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--dump-dir", type=str, default="")
    parser.add_argument("--causal", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    input_dir = args.input_dir
    causal = bool(args.causal)

    # Fixed shapes based on your info
    q_shape = (1, 128, 32, 128)   # [B, S, H_q, D]
    kv_shape = (1, 128, 8, 128)   # [B, S, H_kv, D]

    q = load_tensor_from_bin(os.path.join(input_dir, "graph1_0-output__t92.bin"), q_shape)
    k = load_tensor_from_bin(os.path.join(input_dir, "graph1_0-output__t122.bin"), kv_shape)
    v = load_tensor_from_bin(os.path.join(input_dir, "graph1_0-output__t127.bin"), kv_shape)

    print(f"Q: {q.shape}, K: {k.shape}, V: {v.shape}")

    # --- GQA via SDPA without layout conversion ---
    B, S_q, H_q, D = q.shape
    _, S_kv, H_kv, _ = k.shape

    assert H_q % H_kv == 0, "num_heads must be divisible by num_kv_heads"
    group_size = H_q // H_kv

    # Expand K and V along head dim to match Q: [B, S, H_kv, D] -> [B, S, H_q, D]
    # This is semantic expansion, NOT layout change — still [B, S, H, D]
    k_exp = k.repeat_interleave(group_size, dim=2)  # dim=2 is H
    v_exp = v.repeat_interleave(group_size, dim=2)

    # Transpose to [B, H, S, D] ONLY for SDPA (PyTorch requirement)
    q_sdpa = q.transpose(1, 2)      # [B, H_q, S_q, D]
    k_sdpa = k_exp.transpose(1, 2)  # [B, H_q, S_kv, D]
    v_sdpa = v_exp.transpose(1, 2)  # [B, H_q, S_kv, D]

    # Causal mask: standard upper-triangular for prefill (S_q == S_kv)
    attn_mask = None
    if causal:
        attn_mask = torch.triu(
            torch.full((S_q, S_kv), float('-inf')), diagonal=1
        )
        attn_mask[105:,:] = float('-inf')
        attn_mask[:,105:] = float('-inf')
    print('S_q', S_q, 'S_kv' ,S_kv)
    print(attn_mask)
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    print('q', q.shape, q.dtype)
    print('k', k.shape, k.dtype)
    print('v', v.shape, v.dtype)
    # Compute attention
    with torch.no_grad():
        out_sdpa = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )  # [B, H_q, S_q, D]

    # Transpose back to [B, S_q, H_q, D] — original layout
    output = out_sdpa.transpose(1, 2).contiguous()  # [B, S, H, D]

    print("Done. Output shape:", output.shape)
    print(f"First element: {output[0,0,0,0].item():.9g}")

    # Save output as raw .bin + meta
    if args.dump_dir:
        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
        out_flat = output.cpu().numpy().flatten().astype('<f4')

        bin_path = os.path.join(args.dump_dir, "attn-output.bin")
        with open(bin_path, 'wb') as f:
            f.write(out_flat.tobytes())

        meta_path = os.path.join(args.dump_dir, "attn-output.meta.txt")
        shape = list(output.shape)
        with open(meta_path, 'w') as f:
            f.write("name=attn-output\n")
            f.write("dtype=f32\n")
            f.write(f"rank={len(shape)}\n")
            f.write("shape=" + ",".join(map(str, shape)) + "\n")
            f.write(f"bytes={out_flat.size * 4}\n")

        print(f"Saved to {args.dump_dir}")

if __name__ == "__main__":
    main()
