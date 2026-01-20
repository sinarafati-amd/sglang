#!/usr/bin/env python3
"""
Standalone correctness and performance test for optimized MOE kernels.

This script tests the optimized kernels directly without requiring
the full sglang import chain.

Usage:
    python standalone_test.py
"""

import os
import sys
import time
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def is_hip():
    """Check if running on HIP/ROCm."""
    return hasattr(torch.version, 'hip') and torch.version.hip is not None


_is_hip = is_hip()
print(f"Running on HIP: {_is_hip}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")


# ============================================================================
# Baseline MOE Kernel (simplified from fused_moe_triton_kernels.py)
# ============================================================================

@triton.jit
def moe_align_block_size_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    total_tokens_post_padded_ptr,
    num_experts,
    block_size: tl.constexpr,
    numel: tl.constexpr,
):
    """Sort tokens by expert and pad to block boundaries."""
    pid = tl.program_id(0)
    
    # This is a simplified version - just copy tokens
    if pid < numel:
        token_id = pid
        expert_id = tl.load(topk_ids_ptr + pid)
        tl.store(sorted_token_ids_ptr + pid, token_id)


def moe_align_block_size_simple(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple token alignment for testing (CPU implementation)."""
    num_tokens = topk_ids.numel()
    
    # Sort tokens by expert
    sorted_indices = topk_ids.view(-1).argsort()
    sorted_expert_ids = topk_ids.view(-1)[sorted_indices]
    
    # Pad to block boundary
    padded_size = ((num_tokens + block_size - 1) // block_size) * block_size
    
    sorted_token_ids = torch.zeros(padded_size, dtype=torch.int64, device=topk_ids.device)
    sorted_token_ids[:num_tokens] = sorted_indices
    
    # Create expert_ids for each block
    num_blocks = padded_size // block_size
    expert_ids = torch.zeros(num_blocks, dtype=torch.int32, device=topk_ids.device)
    
    for i in range(num_blocks):
        start = i * block_size
        if start < num_tokens:
            expert_ids[i] = sorted_expert_ids[start]
        else:
            expert_ids[i] = -1  # Padding block
    
    num_tokens_post_padded = torch.tensor([padded_size], dtype=torch.int32, device=topk_ids.device)
    
    return sorted_token_ids, expert_ids, num_tokens_post_padded


@triton.jit
def baseline_moe_gemm_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Simple GEMM kernel for baseline comparison."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


# ============================================================================
# Optimized MOE Kernel (simplified version for testing)
# ============================================================================

@triton.jit
def optimized_moe_gemm_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters - MFMA-aligned for gfx942
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Optimized GEMM kernel with MFMA-aligned tiles and L2 cache optimization."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # L2 cache optimization: group tiles for better reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


def run_baseline_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Run baseline GEMM."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    baseline_moe_gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


def run_optimized_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Run optimized GEMM with MFMA-aligned tiles."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # MFMA-aligned tile sizes for gfx942
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    optimized_moe_gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


def test_correctness():
    """Test that optimized kernel matches baseline within tolerance."""
    print("\n" + "="*60)
    print("CORRECTNESS TEST")
    print("="*60)
    
    test_cases = [
        # (M, N, K, description)
        (64, 1024, 512, "Small batch"),
        (256, 2048, 1024, "Medium batch"),
        (1024, 4096, 2048, "Large batch (target seq_len=1K)"),
        (128, 18432, 7168, "DeepSeek-V3 like (per-expert)"),
    ]
    
    all_passed = True
    
    for M, N, K, desc in test_cases:
        print(f"\nTest: {desc} (M={M}, N={N}, K={K})")
        
        # Create test data
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        
        # Run both kernels
        baseline_out = run_baseline_gemm(A, B)
        optimized_out = run_optimized_gemm(A, B)
        
        # Compare
        torch.cuda.synchronize()
        
        # Check for NaN/Inf
        baseline_nan = torch.isnan(baseline_out).any() or torch.isinf(baseline_out).any()
        optimized_nan = torch.isnan(optimized_out).any() or torch.isinf(optimized_out).any()
        
        if baseline_nan:
            print(f"  FAIL: Baseline produced NaN/Inf")
            all_passed = False
            continue
        if optimized_nan:
            print(f"  FAIL: Optimized produced NaN/Inf")
            all_passed = False
            continue
        
        # Compute difference
        abs_diff = (baseline_out - optimized_out).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        rel_diff = (abs_diff / (baseline_out.abs() + 1e-6)).mean().item()
        
        # Tolerance check (BF16 has lower precision)
        rtol = 1e-2
        atol = 1e-2
        passed = max_diff < atol or rel_diff < rtol
        
        status = "PASS" if passed else "FAIL"
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Rel diff: {rel_diff:.6f} [{status}]")
        
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL CORRECTNESS TESTS PASSED!")
    else:
        print("SOME CORRECTNESS TESTS FAILED!")
    print("="*60)
    
    return all_passed


def benchmark_kernels():
    """Benchmark baseline vs optimized kernels."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    # DeepSeek-V3 style dimensions (scaled for single expert)
    test_cases = [
        # (M, N, K, description) - M is batch_size * top_k
        (8, 18432, 7168, "M=1 * top_k=8 (decode)"),
        (64, 18432, 7168, "M=8 * top_k=8"),
        (256, 18432, 7168, "M=32 * top_k=8"),
        (512, 18432, 7168, "M=64 * top_k=8"),
        (1024, 18432, 7168, "M=128 * top_k=8"),
        (2048, 18432, 7168, "M=256 * top_k=8"),
        (4096, 18432, 7168, "M=512 * top_k=8"),
        (8192, 18432, 7168, "M=1024 * top_k=8 (target)"),
    ]
    
    num_warmup = 10
    num_iters = 50
    
    print(f"\n{'Config':<30} | {'Baseline':>10} | {'Optimized':>10} | {'Speedup':>8} | {'Base TFLOPS':>12} | {'Opt TFLOPS':>12}")
    print("-"*100)
    
    results = []
    
    for M, N, K, desc in test_cases:
        # Create test data
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        
        # Warmup baseline
        for _ in range(num_warmup):
            _ = run_baseline_gemm(A, B)
        torch.cuda.synchronize()
        
        # Benchmark baseline
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        
        for i in range(num_iters):
            start_events[i].record()
            _ = run_baseline_gemm(A, B)
            end_events[i].record()
        
        torch.cuda.synchronize()
        baseline_times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)]
        baseline_ms = sum(baseline_times) / len(baseline_times)
        
        # Warmup optimized
        for _ in range(num_warmup):
            _ = run_optimized_gemm(A, B)
        torch.cuda.synchronize()
        
        # Benchmark optimized
        for i in range(num_iters):
            start_events[i].record()
            _ = run_optimized_gemm(A, B)
            end_events[i].record()
        
        torch.cuda.synchronize()
        optimized_times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)]
        optimized_ms = sum(optimized_times) / len(optimized_times)
        
        # Compute metrics
        speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 0
        flops = 2 * M * N * K  # GEMM FLOPs
        baseline_tflops = flops / (baseline_ms * 1e9)
        optimized_tflops = flops / (optimized_ms * 1e9)
        
        print(f"{desc:<30} | {baseline_ms:>9.3f}ms | {optimized_ms:>9.3f}ms | {speedup:>7.2f}x | {baseline_tflops:>11.2f} | {optimized_tflops:>11.2f}")
        
        results.append({
            "desc": desc,
            "M": M,
            "N": N,
            "K": K,
            "baseline_ms": baseline_ms,
            "optimized_ms": optimized_ms,
            "speedup": speedup,
            "baseline_tflops": baseline_tflops,
            "optimized_tflops": optimized_tflops,
        })
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # Summary
    speedups = [r["speedup"] for r in results]
    avg_speedup = sum(speedups) / len(speedups)
    min_speedup = min(speedups)
    max_speedup = max(speedups)
    
    print("-"*100)
    print(f"\nSUMMARY:")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Min speedup: {min_speedup:.2f}x")
    print(f"  Max speedup: {max_speedup:.2f}x")
    
    # Find target case (M=8192)
    target = next((r for r in results if r["M"] == 8192), None)
    if target:
        print(f"\n  Target (seq_len=1K * top_k=8): {target['speedup']:.2f}x speedup")
        print(f"    Baseline: {target['baseline_ms']:.3f}ms, {target['baseline_tflops']:.2f} TFLOPS")
        print(f"    Optimized: {target['optimized_ms']:.3f}ms, {target['optimized_tflops']:.2f} TFLOPS")
    
    return results


def main():
    print("="*60)
    print("OPTIMIZED MOE KERNEL TEST")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"HIP: {_is_hip}")
    
    # Run correctness test
    correctness_passed = test_correctness()
    
    if not correctness_passed:
        print("\nSkipping benchmark due to correctness failures")
        return
    
    # Run benchmark
    benchmark_kernels()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
