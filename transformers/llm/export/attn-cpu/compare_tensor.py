import numpy as np
import os

def load_tensor_bin(path, expected_numel=524288):
    """Load raw float32 .bin file into numpy array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    if data.size != expected_numel:
        print(f"Warning: {path} has {data.size} elements, expected {expected_numel}")
    return data.flatten()

def cosine_similarity(a, b):
    """Compute cosine similarity between two flat vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def compare_pair(name_a, a, name_b, b):
    print(f"\n=== Comparing {name_a} vs {name_b} ===")
    cos_sim = cosine_similarity(a, b)
    abs_diff = np.abs(a - b)
    max_abs = np.max(abs_diff)
    rmse = np.sqrt(np.mean(abs_diff ** 2))
    mae = np.mean(abs_diff)

    print(f"Cosine Similarity: {cos_sim:.9f}")
    print(f"Max Abs Diff     : {max_abs:.9e}")
    print(f"RMSE             : {rmse:.9e}")
    print(f"MAE              : {mae:.9e}")

def main():
    # Paths
    torch_path = "/home/reck/llama.cpp-test/py_attn_output/attn-output.bin"
    llama_path = "/home/reck/llama.cpp-test/llama-prefill-compare/decode-token-0000/graph1_1-input__t129.bin"
    mnn_path = "/home/reck/mnn_qwen3/mnn-prefill-compare/prefill/graph1_1-run-0000/graph1_1-input__t129.bin"

    # Load tensors (assuming 524288 elements = 128*32*128)
    torch_out = load_tensor_bin(torch_path)
    llama_out = load_tensor_bin(llama_path)
    mnn_out = load_tensor_bin(mnn_path)

    # Pairwise comparisons
    compare_pair("Torch", torch_out, "llama.cpp", llama_out)
    compare_pair("Torch", torch_out, "MNN", mnn_out)
    compare_pair("llama.cpp", llama_out, "MNN", mnn_out)

if __name__ == "__main__":
    main()
