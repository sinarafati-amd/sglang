# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""
Optimized MOE kernels for AMD MI300X (gfx942) and MI350 (gfx950).

This module provides optimized Triton kernels for Mixture of Experts (MOE) layers,
specifically tuned for:
- FP8 (e4m3fnuz) weights and activations
- FP4 (MXFP4) weights with FP8 compute

Key optimizations:
1. MFMA-aligned tile sizes (128x128x64, 256x128x64)
2. Software pipelining with 3-4 stages
3. Fused quantization + GEMM
4. LDS-optimized weight prefetching
5. Wave-level shuffle reductions

Usage:
    Set SGLANG_USE_OPTIMIZED_MOE=1 to enable optimized kernels.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING

from sglang.srt.utils import get_device_name, is_hip

if TYPE_CHECKING:
    from typing import Optional


_is_hip = is_hip()


@functools.lru_cache(maxsize=1)
def get_gpu_arch() -> str:
    """Detect GPU architecture (gfx942, gfx950, etc.)."""
    if not _is_hip:
        return "unknown"
    
    device_name = get_device_name()
    if device_name is None:
        return "unknown"
    
    # Check for MI300X/MI300A (gfx942)
    if "MI300" in device_name or "gfx942" in device_name.lower():
        return "gfx942"
    # Check for MI350 (gfx950)
    if "MI350" in device_name or "gfx950" in device_name.lower():
        return "gfx950"
    
    return "unknown"


def use_optimized_moe_kernel() -> bool:
    """Check if optimized MOE kernel should be used.
    
    Returns True if:
    1. SGLANG_USE_OPTIMIZED_MOE=1 is set
    2. Running on supported AMD GPU (gfx942 or gfx950)
    3. Running on HIP/ROCm
    """
    if not _is_hip:
        return False
    
    env_enabled = os.getenv("SGLANG_USE_OPTIMIZED_MOE", "0") == "1"
    if not env_enabled:
        return False
    
    arch = get_gpu_arch()
    return arch in ("gfx942", "gfx950")


def get_optimized_config(
    M: int,
    E: int,
    N: int,
    K: int,
    dtype_str: str,
    block_shape: Optional[list] = None,
):
    """Get optimized configuration for the given problem size.
    
    Args:
        M: Number of tokens
        E: Number of experts
        N: Output dimension
        K: Input dimension
        dtype_str: Data type string (fp8_w8a8, fp4_w4a8, etc.)
        block_shape: Optional block shape for block-wise quantization
    
    Returns:
        Configuration dictionary for the optimized kernel
    """
    arch = get_gpu_arch()
    
    if arch == "gfx942":
        from .config_gfx942 import get_config_gfx942
        return get_config_gfx942(M, E, N, K, dtype_str, block_shape)
    elif arch == "gfx950":
        from .config_gfx950 import get_config_gfx950
        return get_config_gfx950(M, E, N, K, dtype_str, block_shape)
    else:
        # Fallback to baseline config
        return None


# Lazy imports to avoid circular dependencies
def invoke_fused_moe_kernel_optimized(*args, **kwargs):
    """Invoke the optimized fused MOE kernel."""
    from .fused_moe_optimized import invoke_fused_moe_kernel_optimized as _invoke
    return _invoke(*args, **kwargs)


def fused_experts_optimized(*args, **kwargs):
    """Run fused experts with optimized kernel."""
    from .fused_moe_optimized import fused_experts_optimized as _fused
    return _fused(*args, **kwargs)


__all__ = [
    "use_optimized_moe_kernel",
    "get_optimized_config",
    "get_gpu_arch",
    "invoke_fused_moe_kernel_optimized",
    "fused_experts_optimized",
]
