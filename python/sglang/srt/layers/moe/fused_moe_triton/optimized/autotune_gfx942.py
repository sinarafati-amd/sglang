#!/usr/bin/env python3
# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""
Autotuning script for optimized MOE kernels on AMD MI300X (gfx942).

This script performs exhaustive grid search to find optimal kernel configurations
for different problem sizes and data types.

Usage:
    # Full autotuning (takes ~1-2 hours)
    python autotune_gfx942.py --model deepseek-v3 --dtype fp8 --full
    
    # Quick autotuning (10-15 minutes)
    python autotune_gfx942.py --model deepseek-v3 --dtype fp8 --quick
    
    # Tune specific batch sizes
    python autotune_gfx942.py --batch-sizes 1,64,256,1024 --dtype fp8

Output:
    Updated config_gfx942.py with best configurations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe as fused_moe_baseline
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_hip

_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()


@dataclass
class TuningResult:
    """Result of autotuning for a specific configuration."""
    M: int
    dtype_str: str
    config: Dict[str, Any]
    time_ms: float
    tflops: float
    correct: bool


# Configuration search space for gfx942
# Optimized for MFMA instructions: mfma_f32_32x32x16_fp8, mfma_f32_16x16x32_bf16
SEARCH_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
    "BLOCK_SIZE_N": [64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128],
    "GROUP_SIZE_M": [1, 4, 8, 16, 32],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4],
}

# Reduced search space for quick tuning
QUICK_SEARCH_SPACE = {
    "BLOCK_SIZE_M": [32, 64, 128],
    "BLOCK_SIZE_N": [128, 256],
    "BLOCK_SIZE_K": [64, 128],
    "GROUP_SIZE_M": [8, 16],
    "num_warps": [4, 8],
    "num_stages": [3],
}


def prune_configs(
    M: int, 
    N: int, 
    K: int, 
    configs: List[Dict[str, Any]],
    dtype_str: str,
) -> List[Dict[str, Any]]:
    """Prune invalid or suboptimal configurations."""
    pruned = []
    
    for config in configs:
        bm = config["BLOCK_SIZE_M"]
        bn = config["BLOCK_SIZE_N"]
        bk = config["BLOCK_SIZE_K"]
        gm = config["GROUP_SIZE_M"]
        nw = config["num_warps"]
        ns = config["num_stages"]
        
        # Skip if block sizes don't divide problem dimensions well
        if M > 0 and M < bm and bm > 16:
            continue
        if N > 0 and N % bn != 0:
            continue
        if K > 0 and K % bk != 0:
            continue
            
        # Skip large GROUP_SIZE_M for small M
        if M > 0 and gm * bm > M * 2 and gm > 1:
            continue
        
        # Skip high num_stages with large tiles (register pressure)
        tile_size = bm * bn
        if tile_size >= 32768 and ns > 3:
            continue
        
        # LDS constraint: ~64KB per CU
        # Approximate LDS usage for double buffering
        lds_a = bm * bk * 2  # FP16/BF16
        lds_b = bk * bn * 2
        if dtype_str == "fp8_w8a8":
            lds_b = bk * bn  # FP8
        total_lds = (lds_a + lds_b) * ns
        if total_lds > 65536:
            continue
        
        # Skip low occupancy configs
        threads_per_block = nw * 64  # wavefront size = 64
        if threads_per_block < 256 and tile_size > 8192:
            continue
        
        pruned.append(config)
    
    return pruned


def generate_configs(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all configurations from search space."""
    from itertools import product
    
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    
    configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    
    return configs


def benchmark_config(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output,
    config: Dict[str, Any],
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    num_warmup: int = 5,
    num_iters: int = 20,
) -> Tuple[float, bool]:
    """Benchmark a specific configuration."""
    from .fused_moe_optimized_kernels import invoke_fused_moe_kernel_optimized_fp8
    from ..moe_align_block_size import moe_align_block_size
    import triton.language as tl
    
    try:
        E, N, K = w1.shape
        M = a.shape[0]
        top_k = topk_output.topk_ids.shape[1]
        
        # Align tokens to blocks
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_output.topk_ids, config["BLOCK_SIZE_M"], E
        )
        
        # Prepare output
        total_tokens = sorted_token_ids.shape[0]
        C = torch.empty((total_tokens, N), device=a.device, dtype=a.dtype)
        
        compute_type = tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float16
        
        # Warmup
        for _ in range(num_warmup):
            invoke_fused_moe_kernel_optimized_fp8(
                a, w1, None, C,
                a1_scale, w1_scale,
                topk_output.topk_weights,
                topk_output.topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,  # mul_routed_weight
                top_k,
                config,
                compute_type,
                per_channel_quant=False,
                block_shape=None,
                c_sorted=False,
                filter_expert=True,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        
        for i in range(num_iters):
            start_events[i].record()
            invoke_fused_moe_kernel_optimized_fp8(
                a, w1, None, C,
                a1_scale, w1_scale,
                topk_output.topk_weights,
                topk_output.topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,
                top_k,
                config,
                compute_type,
                per_channel_quant=False,
                block_shape=None,
                c_sorted=False,
                filter_expert=True,
            )
            end_events[i].record()
        
        torch.cuda.synchronize()
        
        times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)]
        avg_time = sum(times) / len(times)
        
        # Check for correctness (no NaN/Inf)
        correct = not (torch.isnan(C).any() or torch.isinf(C).any())
        
        return avg_time, correct
        
    except Exception as e:
        print(f"    Config failed: {e}")
        return float('inf'), False


def autotune_batch_size(
    M: int,
    config_model: Dict[str, Any],
    dtype_str: str,
    search_space: Dict[str, List[Any]],
) -> Optional[TuningResult]:
    """Autotune for a specific batch size."""
    E = config_model["num_experts"]
    N = config_model["intermediate_size"]
    K = config_model["hidden_size"]
    top_k = config_model["top_k"]
    
    use_fp8 = dtype_str == "fp8_w8a8"
    
    # Create test tensors
    dtype = torch.bfloat16
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    w1 = torch.randn(E, 2 * N, K, dtype=dtype, device="cuda")
    score = torch.randn(M, E, dtype=torch.float32, device="cuda")
    
    if use_fp8:
        w1 = w1.to(torch.float8_e4m3fn)
        w1_scale = torch.randn(E, dtype=torch.float32, device="cuda").abs() + 0.1
        a1_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        if _is_fp8_fnuz:
            w1, w1_scale, _ = normalize_e4m3fn_to_e4m3fnuz(w1, w1_scale, a1_scale)
    else:
        w1_scale = None
        a1_scale = None
    
    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=top_k, renormalize=False),
    )
    
    # Generate and prune configs
    all_configs = generate_configs(search_space)
    pruned_configs = prune_configs(M, 2 * N, K, all_configs, dtype_str)
    
    print(f"  Testing {len(pruned_configs)} configs for M={M}...")
    
    best_config = None
    best_time = float('inf')
    
    for i, config in enumerate(pruned_configs):
        if i % 10 == 0:
            print(f"    Progress: {i}/{len(pruned_configs)}")
        
        time_ms, correct = benchmark_config(
            a, w1, None, topk_output, config,
            use_fp8=use_fp8,
            w1_scale=w1_scale,
            a1_scale=a1_scale,
        )
        
        if correct and time_ms < best_time:
            best_time = time_ms
            best_config = config.copy()
    
    if best_config is None:
        print(f"  No valid config found for M={M}")
        return None
    
    # Compute TFLOPS
    flops = 2 * M * top_k * K * 2 * N
    tflops = flops / (best_time * 1e9)
    
    print(f"  Best for M={M}: {best_config}, time={best_time:.3f}ms, {tflops:.2f} TFLOPS")
    
    return TuningResult(
        M=M,
        dtype_str=dtype_str,
        config=best_config,
        time_ms=best_time,
        tflops=tflops,
        correct=True,
    )


def save_tuning_results(results: List[TuningResult], output_path: str):
    """Save tuning results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "arch": "gfx942",
        "results": [
            {
                "M": r.M,
                "dtype": r.dtype_str,
                "config": r.config,
                "time_ms": r.time_ms,
                "tflops": r.tflops,
            }
            for r in results
        ],
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def update_config_file(results: List[TuningResult], dtype_str: str):
    """Update config_gfx942.py with tuning results."""
    from .config_gfx942 import update_config_gfx942
    
    for r in results:
        update_config_gfx942(r.dtype_str, r.M, r.config)
    
    print(f"Updated config_gfx942.py with {len(results)} configurations")


def main():
    parser = argparse.ArgumentParser(description="Autotune MOE kernels for gfx942")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3-small",
        help="Model config name"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp8_w8a8",
        choices=["bf16", "fp8_w8a8", "fp4_w4a8"],
        help="Data type to tune"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,16,32,64,128,256,512,1024",
        help="Comma-separated batch sizes to tune"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced search space for faster tuning"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full search space"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tuning_results_gfx942.json",
        help="Output JSON file"
    )
    args = parser.parse_args()
    
    if not _is_hip:
        print("ERROR: Must run on AMD GPU (HIP/ROCm)")
        sys.exit(1)
    
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    
    # Model configs
    MODEL_CONFIGS = {
        "deepseek-v3": {
            "num_experts": 256,
            "top_k": 8,
            "hidden_size": 7168,
            "intermediate_size": 18432,
        },
        "deepseek-v3-small": {
            "num_experts": 32,
            "top_k": 4,
            "hidden_size": 1792,
            "intermediate_size": 4608,
        },
    }
    
    config_model = MODEL_CONFIGS.get(args.model, MODEL_CONFIGS["deepseek-v3-small"])
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    search_space = QUICK_SEARCH_SPACE if args.quick else SEARCH_SPACE
    
    print(f"Autotuning for gfx942")
    print(f"Model: {args.model}")
    print(f"Dtype: {args.dtype}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Search space size: {len(generate_configs(search_space))}")
    print()
    
    results = []
    for M in batch_sizes:
        result = autotune_batch_size(M, config_model, args.dtype, search_space)
        if result:
            results.append(result)
        torch.cuda.empty_cache()
    
    # Save results
    save_tuning_results(results, args.output)
    
    # Update config file
    update_config_file(results, args.dtype)
    
    print("\nAutotuning complete!")


if __name__ == "__main__":
    main()
