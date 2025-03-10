import time
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


DEFAULT_CONFIG = {
    "batch_size": 10,
    "seq_len": 10,
    "embed_dim": 10,
    "eps": 1e-5,
    "benchmark_size": [2, 10, 1000],
    "num_runs": 100,
    "warmup_runs": 10,
}


def load_extensions():
    """Load CUDA extensions for testing."""
    sources = ["bind.cpp", "layer_norm_kernel.cu", "rms_norm_kernel.cu"]
    functions = load("functions", sources=sources, verbose=True)

    return functions


def get_normalizations(functions):
    """Get dictionary of normalization functions organized by type."""
    norm_funcs = {
        "layer_norm": {
            "cuda": functions.layerNorm,
            "pytorch": lambda x, eps: F.layer_norm(
                x,
                [x.size(-1)],
                eps=eps,
            ),
        },
        "rms_norm": {
            "cuda": functions.rmsNorm,
            "pytorch": lambda x, eps: torch.nn.modules.normalization.RMSNorm(
                x.size(-1), eps=eps
            ).to(x.device)(x),
        },
    }

    return norm_funcs


def benchmark_norm(norm_func, x, eps, num_runs=100, warmup_runs=10):
    """Benchmark a normalization function."""
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup.
    for _ in range(warmup_runs):
        _ = norm_func(x, eps)

    # Benchmark.
    start_event.record()
    for _ in range(num_runs):
        _ = norm_func(x, eps)
    end_event.record()

    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs


def compare_norm_outputs(norm_type, cuda_func, pytorch_func, x, eps):
    """Compare outputs between CUDA and PyTorch implementations."""
    cuda_out = cuda_func(x, eps)
    pytorch_out = pytorch_func(x, eps)

    # Compare results.
    diff = cuda_out - pytorch_out
    max_diff = diff.abs().max().item()
    mean_diff = diff.abs().mean().item()
    is_same = torch.allclose(cuda_out, pytorch_out)

    print(f"\n{norm_type} CUDA vs PyTorch comparison:")
    print(f"Outputs are {'the same' if is_same else 'different'}")
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")

    print("\nDetailed comparison of first few positions:")
    for b in range(min(2, x.size(0))):
        for s in range(min(2, x.size(1))):
            cuda_slice = cuda_out[b, s]
            pytorch_slice = pytorch_out[b, s]
            print(f"Position [{b},{s}]:")
            print(
                f"  CUDA - Mean: {cuda_slice.mean().item():.6f}, Std: {cuda_slice.std().item():.6f}"
            )
            print(
                f"  PyTorch - Mean: {pytorch_slice.mean().item():.6f}, Std: {pytorch_slice.std().item():.6f}"
            )

    return is_same, max_diff, mean_diff


def run_benchmark_suite(norm_funcs, config=None):
    """Run benchmarks for all normalization functions."""
    if config is None:
        config = DEFAULT_CONFIG

    x_large = torch.rand(
        size=config["benchmark_size"], device="cuda", dtype=torch.float32
    )

    print(f"\nSpeed Test (average of {config['num_runs']} runs)\n")
    print(
        f"{'Normalization':<15} | {'CUDA (ms)':<15} | {'PyTorch (ms)':<15} | {'Speedup':<10}"
    )
    print("-" * 65)

    for norm_type, implementations in norm_funcs.items():
        cuda_time = benchmark_norm(
            implementations["cuda"],
            x_large,
            config["eps"],
            config["num_runs"],
            config["warmup_runs"],
        )

        pytorch_time = benchmark_norm(
            implementations["pytorch"],
            x_large,
            config["eps"],
            config["num_runs"],
            config["warmup_runs"],
        )

        speedup = pytorch_time / cuda_time

        print(
            f"{norm_type:<15} | {cuda_time:15.3f} | {pytorch_time:15.3f} | {speedup:5.1f}x"
        )


def test_normalizations():
    """Test different normalization implementations."""
    functions = load_extensions()

    norm_funcs = get_normalizations(functions)

    config = DEFAULT_CONFIG
    x = torch.rand(
        [config["batch_size"], config["seq_len"], config["embed_dim"]],
        device="cuda",
        dtype=torch.float32,
    )

    print(f"Test tensor shape: {x.shape}")

    # Compare each normalization type: CUDA vs PyTorch.
    for norm_type, implementations in norm_funcs.items():
        compare_norm_outputs(
            norm_type,
            implementations["cuda"],
            implementations["pytorch"],
            x,
            config["eps"],
        )

    run_benchmark_suite(norm_funcs, config)


if __name__ == "__main__":
    test_normalizations()
