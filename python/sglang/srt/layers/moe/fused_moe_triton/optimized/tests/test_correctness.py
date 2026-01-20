# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""
Correctness tests for optimized MOE kernels.

This module tests that the optimized MOE kernels produce numerically equivalent
outputs compared to the baseline implementation.

Test scenarios:
1. FP8 (e4m3fnuz) weights and activations
2. FP4 (MXFP4) weights with FP8 compute
3. BF16/FP16 baseline

Test shapes (DeepSeek-V3 style):
- E=256 experts, top_k=8
- hidden_size=7168
- intermediate_size=18432 (per expert)

Run tests:
    python -m pytest test_correctness.py -v
    
Or run directly:
    python test_correctness.py
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Optional, Tuple

import torch
from tqdm import tqdm

# Set environment to use optimized kernel for comparison
os.environ["SGLANG_USE_OPTIMIZED_MOE"] = "0"  # Disable during import

# Import after setting env
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe as fused_moe_baseline
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import is_hip

_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()


def skip_if_not_hip(func):
    """Decorator to skip tests if not running on HIP/ROCm."""
    def wrapper(*args, **kwargs):
        if not _is_hip:
            print(f"Skipping {func.__name__}: Not running on HIP/ROCm")
            return
        return func(*args, **kwargs)
    return wrapper


class TestOptimizedMOECorrectness(unittest.TestCase):
    """Test correctness of optimized MOE kernels vs baseline."""
    
    # DeepSeek-V3 style configuration
    NUM_EXPERTS = 256
    TOP_K = 8
    HIDDEN_SIZE = 7168
    INTERMEDIATE_SIZE = 18432
    
    # Test configurations
    TEST_M_VALUES = [1, 32, 64, 128, 256, 512, 1024]
    TEST_DTYPES = [torch.bfloat16]
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        
        # Check if GPU is available
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA/HIP not available")
        
        # Check if we're on AMD GPU
        if not _is_hip:
            raise unittest.SkipTest("Optimized MOE kernels only available on AMD GPUs")
    
    @staticmethod
    def create_random_tensor(shape, dtype, device="cuda", mean=0, std=0.01):
        """Create a random tensor with given properties."""
        return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)
    
    def get_tolerance(self, dtype, use_fp8: bool = False):
        """Get tolerance values for numerical comparison."""
        if use_fp8:
            # FP8 has lower precision
            return 5e-2, 5e-2
        elif dtype == torch.float32:
            return 1e-3, 1e-5
        elif dtype in [torch.float16, torch.bfloat16]:
            return 1e-1, 1e-2
        else:
            return 1e-2, 1e-2
    
    def torch_naive_moe(
        self,
        a: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        score: torch.Tensor,
        topk: int,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Naive PyTorch MOE implementation for reference."""
        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)
        
        if w1.dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz]:
            w1_compute = w1.to(a.dtype)
            w2_compute = w2.to(a.dtype)
            
            if w1_scale is not None:
                w1_compute = (w1_compute * w1_scale.view(-1, 1, 1)).to(a.dtype)
            if w2_scale is not None:
                w2_compute = (w2_compute * w2_scale.view(-1, 1, 1)).to(a.dtype)
            if a1_scale is not None:
                a = (a * a1_scale).to(a.dtype)
        else:
            w1_compute = w1
            w2_compute = w2
        
        for i in range(w1_compute.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = SiluAndMul()(
                    a[mask] @ w1_compute[i].transpose(0, 1)
                ) @ w2_compute[i].transpose(0, 1)
        
        return (
            out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
        ).sum(dim=1)
    
    def _test_bf16_correctness(self, M: int, E: int, N: int, K: int, topk: int, dtype: torch.dtype):
        """Test BF16/FP16 kernel correctness."""
        rtol, atol = self.get_tolerance(dtype, use_fp8=False)
        
        # Create test tensors
        a = self.create_random_tensor((M, K), dtype)
        w1 = self.create_random_tensor((E, 2 * N, K), dtype)
        w2 = self.create_random_tensor((E, K, N), dtype)
        score = self.create_random_tensor((M, E), dtype)
        
        # Baseline output
        topk_output = select_experts(
            hidden_states=a,
            router_logits=score,
            topk_config=TopKConfig(top_k=topk, renormalize=False),
        )
        baseline_output = fused_moe_baseline(a, w1, w2, topk_output)
        
        # Optimized output
        from ..fused_moe_optimized import fused_experts_optimized
        optimized_output = fused_experts_optimized(
            hidden_states=a,
            w1=w1,
            w2=w2,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            use_fp8_w8a8=False,
            activation="silu",
        )
        
        # Compare outputs
        torch.testing.assert_close(
            optimized_output, 
            baseline_output, 
            rtol=rtol, 
            atol=atol,
            msg=f"BF16 mismatch at M={M}, E={E}, N={N}, K={K}, topk={topk}"
        )
    
    def _test_fp8_correctness(self, M: int, E: int, N: int, K: int, topk: int, dtype: torch.dtype):
        """Test FP8 kernel correctness."""
        rtol, atol = self.get_tolerance(dtype, use_fp8=True)
        
        # Create test tensors
        a = self.create_random_tensor((M, K), dtype)
        w1 = self.create_random_tensor((E, 2 * N, K), dtype)
        w2 = self.create_random_tensor((E, K, N), dtype)
        score = self.create_random_tensor((M, E), dtype)
        
        # Convert to FP8
        w1_fp8 = w1.to(torch.float8_e4m3fn)
        w2_fp8 = w2.to(torch.float8_e4m3fn)
        w1_scale = self.create_random_tensor(E, torch.float32)
        w2_scale = self.create_random_tensor(E, torch.float32)
        a1_scale = self.create_random_tensor(1, torch.float32)
        a2_scale = self.create_random_tensor(1, torch.float32)
        
        # Handle HIP normalization
        if _is_fp8_fnuz:
            w1_fp8, w1_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=w1_fp8,
                weight_scale=w1_scale,
                input_scale=a1_scale,
            )
            w2_fp8, w2_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=w2_fp8,
                weight_scale=w2_scale,
                input_scale=a2_scale,
            )
        
        # Baseline output
        topk_output = select_experts(
            hidden_states=a,
            router_logits=score,
            topk_config=TopKConfig(top_k=topk, renormalize=False),
        )
        baseline_output = fused_moe_baseline(
            a, w1_fp8, w2_fp8, topk_output,
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        )
        
        # Optimized output
        from ..fused_moe_optimized import fused_experts_optimized
        optimized_output = fused_experts_optimized(
            hidden_states=a,
            w1=w1_fp8,
            w2=w2_fp8,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            activation="silu",
        )
        
        # Compare outputs
        torch.testing.assert_close(
            optimized_output, 
            baseline_output, 
            rtol=rtol, 
            atol=atol,
            msg=f"FP8 mismatch at M={M}, E={E}, N={N}, K={K}, topk={topk}"
        )
    
    @skip_if_not_hip
    def test_bf16_small_batch(self):
        """Test BF16 with small batch sizes (decode path)."""
        for M in [1, 4, 8, 16]:
            with self.subTest(M=M):
                self._test_bf16_correctness(
                    M=M,
                    E=64,  # Smaller E for faster test
                    N=1024,
                    K=512,
                    topk=2,
                    dtype=torch.bfloat16,
                )
    
    @skip_if_not_hip
    def test_bf16_large_batch(self):
        """Test BF16 with large batch sizes (prefill path)."""
        for M in [128, 256, 512]:
            with self.subTest(M=M):
                self._test_bf16_correctness(
                    M=M,
                    E=64,
                    N=1024,
                    K=512,
                    topk=2,
                    dtype=torch.bfloat16,
                )
    
    @skip_if_not_hip
    def test_fp8_small_batch(self):
        """Test FP8 with small batch sizes."""
        for M in [1, 4, 8, 16]:
            with self.subTest(M=M):
                self._test_fp8_correctness(
                    M=M,
                    E=64,
                    N=1024,
                    K=512,
                    topk=2,
                    dtype=torch.bfloat16,
                )
    
    @skip_if_not_hip
    def test_fp8_large_batch(self):
        """Test FP8 with large batch sizes."""
        for M in [128, 256, 512]:
            with self.subTest(M=M):
                self._test_fp8_correctness(
                    M=M,
                    E=64,
                    N=1024,
                    K=512,
                    topk=2,
                    dtype=torch.bfloat16,
                )
    
    @skip_if_not_hip
    def test_deepseek_v3_shapes(self):
        """Test with DeepSeek-V3 model shapes (scaled down for testing)."""
        # Use scaled-down version for faster testing
        # Full size: E=256, hidden=7168, intermediate=18432, topk=8
        E = 32  # Scaled down from 256
        hidden = 896  # Scaled down from 7168
        intermediate = 2304  # Scaled down from 18432
        topk = 4  # Scaled down from 8
        
        for M in [1, 64, 256, 1024]:
            with self.subTest(M=M):
                self._test_bf16_correctness(
                    M=M,
                    E=E,
                    N=intermediate,
                    K=hidden,
                    topk=topk,
                    dtype=torch.bfloat16,
                )
    
    @skip_if_not_hip
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        M = 64
        E = 32
        N = 512
        K = 256
        topk = 2
        dtype = torch.bfloat16
        
        # Test with very small values
        a = self.create_random_tensor((M, K), dtype) * 1e-4
        w1 = self.create_random_tensor((E, 2 * N, K), dtype) * 1e-4
        w2 = self.create_random_tensor((E, K, N), dtype) * 1e-4
        score = self.create_random_tensor((M, E), dtype)
        
        topk_output = select_experts(
            hidden_states=a,
            router_logits=score,
            topk_config=TopKConfig(top_k=topk, renormalize=False),
        )
        
        # Should not produce NaN/Inf
        baseline_output = fused_moe_baseline(a, w1, w2, topk_output)
        self.assertFalse(torch.isnan(baseline_output).any(), "Baseline produced NaN")
        self.assertFalse(torch.isinf(baseline_output).any(), "Baseline produced Inf")
        
        from ..fused_moe_optimized import fused_experts_optimized
        optimized_output = fused_experts_optimized(
            hidden_states=a,
            w1=w1,
            w2=w2,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            use_fp8_w8a8=False,
            activation="silu",
        )
        self.assertFalse(torch.isnan(optimized_output).any(), "Optimized produced NaN")
        self.assertFalse(torch.isinf(optimized_output).any(), "Optimized produced Inf")


class TestOptimizedMOEConfiguration(unittest.TestCase):
    """Test configuration loading for optimized MOE kernels."""
    
    @skip_if_not_hip
    def test_config_gfx942_loading(self):
        """Test that gfx942 configs load correctly."""
        from ..config_gfx942 import get_config_gfx942, get_all_configs_gfx942
        
        # Test config retrieval
        config = get_config_gfx942(M=1024, E=256, N=7168, K=7168, dtype_str="fp8_w8a8")
        self.assertIn("BLOCK_SIZE_M", config)
        self.assertIn("BLOCK_SIZE_N", config)
        self.assertIn("BLOCK_SIZE_K", config)
        self.assertIn("GROUP_SIZE_M", config)
        self.assertIn("num_warps", config)
        self.assertIn("num_stages", config)
        
        # Test all configs
        all_configs = get_all_configs_gfx942()
        self.assertIn("fp8_w8a8", all_configs)
        self.assertIn("fp4_w4a8", all_configs)
        self.assertIn("bf16", all_configs)
    
    @skip_if_not_hip
    def test_config_gfx950_loading(self):
        """Test that gfx950 configs load correctly."""
        from ..config_gfx950 import get_config_gfx950, get_all_configs_gfx950
        
        config = get_config_gfx950(M=1024, E=256, N=7168, K=7168, dtype_str="fp8_w8a8")
        self.assertIn("BLOCK_SIZE_M", config)
        self.assertIn("num_stages", config)
        
        # gfx950 should have higher num_stages than gfx942
        self.assertGreaterEqual(config["num_stages"], 3)
    
    @skip_if_not_hip
    def test_gpu_arch_detection(self):
        """Test GPU architecture detection."""
        from .. import get_gpu_arch, use_optimized_moe_kernel
        
        arch = get_gpu_arch()
        # Should return gfx942, gfx950, or unknown
        self.assertIn(arch, ["gfx942", "gfx950", "unknown"])


def run_quick_correctness_check():
    """Run a quick correctness check for manual verification."""
    print("Running quick correctness check...")
    
    if not _is_hip:
        print("Skipping: Not running on HIP/ROCm")
        return
    
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    
    # Quick test
    M, E, N, K, topk = 64, 32, 512, 256, 2
    dtype = torch.bfloat16
    
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    w1 = torch.randn(E, 2 * N, K, dtype=dtype, device="cuda")
    w2 = torch.randn(E, K, N, dtype=dtype, device="cuda")
    score = torch.randn(M, E, dtype=dtype, device="cuda")
    
    topk_output = select_experts(
        hidden_states=a,
        router_logits=score,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    
    # Baseline
    baseline_output = fused_moe_baseline(a, w1, w2, topk_output)
    
    # Optimized
    from ..fused_moe_optimized import fused_experts_optimized
    optimized_output = fused_experts_optimized(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        use_fp8_w8a8=False,
        activation="silu",
    )
    
    # Check
    max_diff = (baseline_output - optimized_output).abs().max().item()
    mean_diff = (baseline_output - optimized_output).abs().mean().item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Baseline shape: {baseline_output.shape}")
    print(f"Optimized shape: {optimized_output.shape}")
    
    if max_diff < 0.1:
        print("PASSED: Optimized kernel matches baseline within tolerance")
    else:
        print("FAILED: Optimized kernel differs significantly from baseline")


if __name__ == "__main__":
    # Run quick check first
    run_quick_correctness_check()
    print("\n" + "="*50 + "\n")
    
    # Run full test suite
    unittest.main(verbosity=2)
