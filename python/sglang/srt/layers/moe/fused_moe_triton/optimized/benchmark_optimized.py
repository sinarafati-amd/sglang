#!/usr/bin/env python3
# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""
A/B Benchmark: Optimized MOE Kernels vs Baseline

This script benchmarks the optimized MOE kernels against the baseline
implementation to measure performance improvements.

Test scenarios:
1. FP8 (e4m3fnuz) with seq length 1K
2. FP4 (MXFP4) with seq length 1K
3. BF16 baseline

Target model: DeepSeek-V3 style (256 experts, top-8, hidden_size=7168)

Usage:
    python benchmark_optimized.py --model deepseek-v3 --dtype fp8
    python benchmark_optimized.py --model deepseek-v3 --dtype fp4
    python benchmark_optimized.py --batch-sizes 1,32,128,512,1024 --dtype fp8

Output:
    JSON results file with detailed timing and speedup metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import triton

# Disable optimized kernel initially for baseline comparison
os.environ["SGLANG_USE_OPTIMIZED_MOE"] = "0"

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe as fused_moe_baseline
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_hip

_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    batch_size: int
    dtype: str
    baseline_ms: float
    optimized_ms: float
    speedup: float
    baseline_tflops: float
    optimized_tflops: float
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    correctness_passed: bool


@dataclass
class ModelConfig:
    """Model configuration for benchmarking."""
    name: str
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    dtype: torch.dtype = torch.bfloat16


# Predefined model configurations
MODEL_CONFIGS = {
    "deepseek-v3": ModelConfig(
        name="DeepSeek-V3",
        num_experts=256,
        top_k=8,
        hidden_size=7168,
        intermediate_size=18432,
    ),
    "deepseek-v3-small": ModelConfig(
        name="DeepSeek-V3 (Scaled)",
        num_experts=32,  # Scaled down for faster testing
        top_k=4,
        hidden_size=1792,
        intermediate_size=4608,
    ),
    "mixtral-8x7b": ModelConfig(
        name="Mixtral-8x7B",
        num_experts=8,
        top_k=2,
        hidden_size=4096,
        intermediate_size=14336,
    ),
    "qwen2-moe": ModelConfig(
        name="Qwen2-MoE",
        num_experts=60,
        top_k=4,
        hidden_size=2048,
        intermediate_size=5632,
    ),
}


def compute_flops(
    M: int,
    E: int,
    N: int,
    K: int,
    top_k: int,
) -> float:
    """Compute theoretical FLOPs for MOE layer."""
    # First GEMM: [M*top_k, K] @ [K, 2*N] -> [M*top_k, 2*N]
    gemm1_flops = 2 * M * top_k * K * 2 * N
    
    # Activation (SiLU) + multiply: [M*top_k, N] * [M*top_k, N]
    act_flops = 3 * M * top_k * N  # Approximate
    
    # Second GEMM: [M*top_k, N] @ [N, K] -> [M*top_k, K]
    gemm2_flops = 2 * M * top_k * N * K
    
    # Sum reduction: [M, top_k, K] -> [M, K]
    reduce_flops = M * top_k * K
    
    return gemm1_flops + act_flops + gemm2_flops + reduce_flops


def create_test_tensors(
    batch_size: int,
    config: ModelConfig,
    use_fp8: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Create test tensors for benchmarking."""
    M = batch_size
    E = config.num_experts
    K = config.hidden_size
    N = config.intermediate_size
    dtype = config.dtype
    
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    w1 = torch.randn(E, 2 * N, K, dtype=dtype, device="cuda")
    w2 = torch.randn(E, K, N, dtype=dtype, device="cuda")
    score = torch.randn(M, E, dtype=torch.float32, device="cuda")
    
    if use_fp8:
        w1_fp8 = w1.to(torch.float8_e4m3fn)
        w2_fp8 = w2.to(torch.float8_e4m3fn)
        w1_scale = torch.randn(E, dtype=torch.float32, device="cuda").abs() + 0.1
        w2_scale = torch.randn(E, dtype=torch.float32, device="cuda").abs() + 0.1
        a1_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        a2_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        
        if _is_fp8_fnuz:
            w1_fp8, w1_scale, _ = normalize_e4m3fn_to_e4m3fnuz(w1_fp8, w1_scale, a1_scale)
            w2_fp8, w2_scale, _ = normalize_e4m3fn_to_e4m3fnuz(w2_fp8, w2_scale, a2_scale)
        
        return a, w1_fp8, w2_fp8, score, w1_scale, w2_scale, a1_scale, a2_scale
    else:
        return a, w1, w2, score, None, None, None, None


def benchmark_baseline(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    top_k: int,
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> Tuple[float, torch.Tensor]:
    """Benchmark baseline MOE kernel."""
    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=top_k, renormalize=False),
    )
    
    # Warmup
    for _ in range(num_warmup):
        if use_fp8:
            out = fused_moe_baseline(
                a, w1, w2, topk_output,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )
        else:
            out = fused_moe_baseline(a, w1, w2, topk_output)
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    
    for i in range(num_iters):
        start_events[i].record()
        if use_fp8:
            out = fused_moe_baseline(
                a, w1, w2, topk_output,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )
        else:
            out = fused_moe_baseline(a, w1, w2, topk_output)
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)]
    avg_time_ms = sum(times) / len(times)
    
    return avg_time_ms, out


def benchmark_optimized(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    top_k: int,
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> Tuple[float, torch.Tensor]:
    """Benchmark optimized MOE kernel."""
    from .fused_moe_optimized import fused_experts_optimized
    
    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=top_k, renormalize=False),
    )
    
    # Warmup
    for _ in range(num_warmup):
        out = fused_experts_optimized(
            hidden_states=a,
            w1=w1,
            w2=w2,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            use_fp8_w8a8=use_fp8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            activation="silu",
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    
    for i in range(num_iters):
        start_events[i].record()
        out = fused_experts_optimized(
            hidden_states=a,
            w1=w1,
            w2=w2,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            use_fp8_w8a8=use_fp8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            activation="silu",
        )
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)]
    avg_time_ms = sum(times) / len(times)
    
    return avg_time_ms, out


def check_correctness(
    baseline_out: torch.Tensor,
    optimized_out: torch.Tensor,
    rtol: float = 1e-1,
    atol: float = 1e-2,
) -> bool:
    """Check if optimized output matches baseline."""
    try:
        torch.testing.assert_close(optimized_out, baseline_out, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        max_diff = (baseline_out - optimized_out).abs().max().item()
        print(f"  Correctness check FAILED: max diff = {max_diff:.6f}")
        return False


def run_benchmark(
    batch_sizes: List[int],
    config: ModelConfig,
    dtype_str: str,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    results = []
    use_fp8 = dtype_str in ["fp8", "fp8_w8a8"]
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {config.name}")
    print(f"  Experts: {config.num_experts}, Top-K: {config.top_k}")
    print(f"  Hidden: {config.hidden_size}, Intermediate: {config.intermediate_size}")
    print(f"  Dtype: {dtype_str}")
    print(f"{'='*60}\n")
    
    print(f"{'Batch':>8} | {'Baseline':>10} | {'Optimized':>10} | {'Speedup':>8} | {'Correct':>8}")
    print(f"{'-'*8} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8}")
    
    for batch_size in batch_sizes:
        # Create tensors
        tensors = create_test_tensors(batch_size, config, use_fp8=use_fp8)
        a, w1, w2, score = tensors[:4]
        w1_scale, w2_scale, a1_scale, a2_scale = tensors[4:]
        
        # Compute theoretical FLOPs
        flops = compute_flops(
            batch_size, 
            config.num_experts, 
            config.intermediate_size, 
            config.hidden_size,
            config.top_k
        )
        
        try:
            # Benchmark baseline
            baseline_ms, baseline_out = benchmark_baseline(
                a, w1, w2, score, config.top_k,
                use_fp8=use_fp8,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                num_warmup=num_warmup,
                num_iters=num_iters,
            )
            
            # Benchmark optimized
            optimized_ms, optimized_out = benchmark_optimized(
                a, w1, w2, score, config.top_k,
                use_fp8=use_fp8,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                num_warmup=num_warmup,
                num_iters=num_iters,
            )
            
            # Check correctness
            correctness = check_correctness(baseline_out, optimized_out)
            
            # Compute metrics
            speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 0
            baseline_tflops = flops / (baseline_ms * 1e9)
            optimized_tflops = flops / (optimized_ms * 1e9)
            
            result = BenchmarkResult(
                batch_size=batch_size,
                dtype=dtype_str,
                baseline_ms=baseline_ms,
                optimized_ms=optimized_ms,
                speedup=speedup,
                baseline_tflops=baseline_tflops,
                optimized_tflops=optimized_tflops,
                num_experts=config.num_experts,
                top_k=config.top_k,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                correctness_passed=correctness,
            )
            results.append(result)
            
            status = "PASS" if correctness else "FAIL"
            print(f"{batch_size:>8} | {baseline_ms:>9.3f}ms | {optimized_ms:>9.3f}ms | {speedup:>7.2f}x | {status:>8}")
            
        except Exception as e:
            print(f"{batch_size:>8} | ERROR: {str(e)[:40]}")
            continue
        
        # Clear cache
        torch.cuda.empty_cache()
    
    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary."""
    if not results:
        print("No results to summarize")
        return
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Average speedup
    speedups = [r.speedup for r in results]
    avg_speedup = sum(speedups) / len(speedups)
    min_speedup = min(speedups)
    max_speedup = max(speedups)
    
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Min speedup: {min_speedup:.2f}x")
    print(f"Max speedup: {max_speedup:.2f}x")
    
    # Correctness
    correct_count = sum(1 for r in results if r.correctness_passed)
    print(f"Correctness: {correct_count}/{len(results)} passed")
    
    # Best case
    best = max(results, key=lambda r: r.speedup)
    print(f"\nBest case: batch_size={best.batch_size}, speedup={best.speedup:.2f}x")
    
    # Throughput comparison
    print(f"\nThroughput comparison (TFLOPS):")
    for r in results:
        print(f"  M={r.batch_size}: baseline={r.baseline_tflops:.2f}, optimized={r.optimized_tflops:.2f}")


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "results": [asdict(r) for r in results],
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark optimized MOE kernels")
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-v3-small",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model configuration to benchmark"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp8", "fp4"],
        help="Data type to benchmark"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,32,64,128,256,512,1024",
        help="Comma-separated list of batch sizes"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=50,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()
    
    if not _is_hip:
        print("ERROR: Optimized MOE kernels only available on AMD GPUs (HIP/ROCm)")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA/HIP not available")
        sys.exit(1)
    
    # Initialize
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    
    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    # Get model config
    config = MODEL_CONFIGS[args.model]
    
    # Run benchmark
    results = run_benchmark(
        batch_sizes=batch_sizes,
        config=config,
        dtype_str=args.dtype,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
