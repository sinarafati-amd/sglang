# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""
Optimized MOE kernel configurations for AMD MI300X (gfx942).

Hardware specifications:
- Architecture: CDNA3 (gfx942)
- Compute Units: 304
- Memory: 192 GB HBM3
- Bandwidth: 5.3 TB/s
- LDS per CU: 64 KB
- Wavefront size: 64

Key MFMA instructions:
- mfma_f32_32x32x16_fp8: FP8 matrix multiply
- mfma_f32_16x16x32_bf16: BF16 matrix multiply
- mfma_f32_32x32x8_f16: FP16 matrix multiply

Optimization strategy:
1. MFMA-aligned block sizes (multiples of 32 for M/N, 16 for K with FP8)
2. Large GROUP_SIZE_M for L2 cache reuse across expert blocks
3. Software pipelining with 3-4 stages for memory latency hiding
4. num_warps tuned for occupancy vs register pressure

Target model: DeepSeek-V3 (256 experts, top-8, hidden_size=7168)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Configuration lookup table indexed by (M_bucket, dtype_str)
# Each entry contains: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages

_FP8_CONFIGS = {
    # M=1: Decode path - small M, optimize for latency
    1: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    },
    # M=16: Small batch decode
    16: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    },
    # M=32: Medium decode batch
    32: {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 3,
    },
    # M=64: Transition to prefill-like
    64: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    },
    # M=128: Prefill path starts
    128: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    # M=256: Medium prefill
    256: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 3,
    },
    # M=512: Larger prefill
    512: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 3,
    },
    # M=1024: Target sequence length (1K)
    1024: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 3,
    },
    # M=2048: Large prefill
    2048: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 3,
    },
    # M=4096: Very large prefill
    4096: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 3,
    },
    # M=8192: Maximum batch
    8192: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 2,  # Reduced due to register pressure
    },
}

_FP4_CONFIGS = {
    # FP4 configs - similar structure but adjusted for FP4->FP8 compute
    # FP4 has 2x compression, so we can afford larger K blocks
    1: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,  # Larger K for FP4
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    },
    16: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    },
    32: {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 3,
    },
    64: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    },
    128: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    256: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 3,
    },
    512: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 3,
    },
    1024: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 3,
    },
    2048: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 3,
    },
    4096: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 3,
    },
    8192: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 2,
    },
}

_BF16_CONFIGS = {
    # BF16/FP16 configs - baseline optimization
    1: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    },
    16: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    },
    32: {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 3,
    },
    64: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    },
    128: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    256: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 3,
    },
    512: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 3,
    },
    1024: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 3,
    },
    2048: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 3,
    },
    4096: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 3,
    },
    8192: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 2,
    },
}

# Block-wise quantization specific configs
_FP8_BLOCKWISE_CONFIGS = {
    # For block_shape=[128, 128] quantization
    1: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,  # Must match block_shape[1]
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    },
    32: {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 3,
    },
    128: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    },
    512: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 3,
    },
    1024: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 3,
    },
    2048: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 3,
    },
    4096: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 2,
    },
}


def _find_nearest_config(M: int, configs: Dict[int, Dict]) -> Dict[str, Any]:
    """Find the config for the nearest M value."""
    keys = sorted(configs.keys())
    
    # Find the closest M
    nearest_m = keys[0]
    min_diff = abs(M - keys[0])
    
    for k in keys:
        diff = abs(M - k)
        if diff < min_diff:
            min_diff = diff
            nearest_m = k
        # If M is between two values, prefer the larger one
        if k <= M:
            nearest_m = k
    
    return configs[nearest_m].copy()


def get_config_gfx942(
    M: int,
    E: int,
    N: int,
    K: int,
    dtype_str: Optional[str] = None,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Get optimized configuration for gfx942 (MI300X).
    
    Args:
        M: Number of tokens (batch size * seq_len for MOE)
        E: Number of experts
        N: Output dimension (intermediate_size for gate_up, hidden_size for down)
        K: Input dimension (hidden_size for gate_up, intermediate_size for down)
        dtype_str: Data type string ("fp8_w8a8", "fp4_w4a8", None for BF16/FP16)
        block_shape: Optional block shape for block-wise quantization [block_n, block_k]
    
    Returns:
        Configuration dictionary with BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_M, num_warps, num_stages
    """
    # Select config table based on dtype
    if dtype_str == "fp8_w8a8":
        if block_shape is not None and block_shape[0] > 0 and block_shape[1] > 0:
            configs = _FP8_BLOCKWISE_CONFIGS
        else:
            configs = _FP8_CONFIGS
    elif dtype_str == "fp4_w4a8":
        configs = _FP4_CONFIGS
    else:
        configs = _BF16_CONFIGS
    
    # Get base config
    config = _find_nearest_config(M, configs)
    
    # Adjust for block-wise quantization constraints
    if block_shape is not None and block_shape[1] > 0:
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        if config["BLOCK_SIZE_K"] % block_shape[1] != 0:
            config["BLOCK_SIZE_K"] = block_shape[1]
    
    # Ensure K dimension alignment
    if K > 0 and K % config["BLOCK_SIZE_K"] != 0:
        # Find a BLOCK_SIZE_K that divides K
        for bk in [32, 64, 128, 256]:
            if K % bk == 0:
                config["BLOCK_SIZE_K"] = bk
                break
    
    # Ensure N dimension alignment
    if N > 0 and N % config["BLOCK_SIZE_N"] != 0:
        for bn in [64, 128, 256]:
            if N % bn == 0:
                config["BLOCK_SIZE_N"] = bn
                break
    
    return config


def get_all_configs_gfx942() -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Return all configuration tables for gfx942."""
    return {
        "fp8_w8a8": _FP8_CONFIGS.copy(),
        "fp4_w4a8": _FP4_CONFIGS.copy(),
        "bf16": _BF16_CONFIGS.copy(),
        "fp8_blockwise": _FP8_BLOCKWISE_CONFIGS.copy(),
    }


def update_config_gfx942(
    dtype_str: str,
    M: int,
    config: Dict[str, Any],
) -> None:
    """
    Update a configuration entry (used by autotuning).
    
    Args:
        dtype_str: Data type string
        M: Token count bucket
        config: New configuration dictionary
    """
    if dtype_str == "fp8_w8a8":
        _FP8_CONFIGS[M] = config
    elif dtype_str == "fp4_w4a8":
        _FP4_CONFIGS[M] = config
    elif dtype_str == "fp8_blockwise":
        _FP8_BLOCKWISE_CONFIGS[M] = config
    else:
        _BF16_CONFIGS[M] = config
