# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""
Optimized MOE kernel configurations for AMD MI350 (gfx950).

Hardware specifications (estimated based on CDNA4 evolution):
- Architecture: CDNA4 (gfx950)
- Expected improvements over gfx942:
  - Higher MFMA throughput
  - Improved memory bandwidth
  - Larger register file (allowing more stages)
  - Enhanced LDS bandwidth

Key optimization differences from gfx942:
1. Higher num_stages (4-5) due to improved register file
2. Potentially larger tile sizes for better MFMA utilization
3. More aggressive software pipelining

Note: These configs are preliminary and should be autotuned on actual hardware.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# gfx950 configurations - adapted from gfx942 with gfx950-specific optimizations
# Main differences:
# - Higher num_stages (4-5 vs 3-4)
# - Larger tiles where register pressure allows
# - More aggressive GROUP_SIZE_M for L2 reuse

_FP8_CONFIGS = {
    1: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,  # Higher than gfx942
    },
    16: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
    },
    32: {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 4,
    },
    64: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    128: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    },
    256: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    },
    512: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    },
    1024: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,  # Larger K for gfx950
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    },
    2048: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 4,
    },
    4096: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
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
        "num_stages": 3,
    },
}

_FP4_CONFIGS = {
    1: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
    },
    16: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
    },
    32: {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 4,
    },
    64: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    128: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    },
    256: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    },
    512: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    },
    1024: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 256,  # Larger K for FP4 on gfx950
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    },
    2048: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 4,
    },
    4096: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
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
        "num_stages": 3,
    },
}

_BF16_CONFIGS = {
    1: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
    },
    16: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
    },
    32: {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 4,
    },
    64: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    128: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    },
    256: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    },
    512: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    },
    1024: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    },
    2048: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 4,
    },
    4096: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
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
        "num_stages": 3,
    },
}

_FP8_BLOCKWISE_CONFIGS = {
    1: {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
    },
    32: {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 4,
    },
    128: {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    },
    512: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    },
    1024: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 4,
    },
    2048: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 4,
    },
    4096: {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 3,
    },
}


def _find_nearest_config(M: int, configs: Dict[int, Dict]) -> Dict[str, Any]:
    """Find the config for the nearest M value."""
    keys = sorted(configs.keys())
    
    nearest_m = keys[0]
    min_diff = abs(M - keys[0])
    
    for k in keys:
        diff = abs(M - k)
        if diff < min_diff:
            min_diff = diff
            nearest_m = k
        if k <= M:
            nearest_m = k
    
    return configs[nearest_m].copy()


def get_config_gfx950(
    M: int,
    E: int,
    N: int,
    K: int,
    dtype_str: Optional[str] = None,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Get optimized configuration for gfx950 (MI350).
    
    Args:
        M: Number of tokens
        E: Number of experts
        N: Output dimension
        K: Input dimension
        dtype_str: Data type string
        block_shape: Optional block shape for block-wise quantization
    
    Returns:
        Configuration dictionary
    """
    if dtype_str == "fp8_w8a8":
        if block_shape is not None and block_shape[0] > 0 and block_shape[1] > 0:
            configs = _FP8_BLOCKWISE_CONFIGS
        else:
            configs = _FP8_CONFIGS
    elif dtype_str == "fp4_w4a8":
        configs = _FP4_CONFIGS
    else:
        configs = _BF16_CONFIGS
    
    config = _find_nearest_config(M, configs)
    
    # Adjust for block-wise quantization constraints
    if block_shape is not None and block_shape[1] > 0:
        if config["BLOCK_SIZE_K"] % block_shape[1] != 0:
            config["BLOCK_SIZE_K"] = block_shape[1]
    
    # Ensure K dimension alignment
    if K > 0 and K % config["BLOCK_SIZE_K"] != 0:
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


def get_all_configs_gfx950() -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Return all configuration tables for gfx950."""
    return {
        "fp8_w8a8": _FP8_CONFIGS.copy(),
        "fp4_w4a8": _FP4_CONFIGS.copy(),
        "bf16": _BF16_CONFIGS.copy(),
        "fp8_blockwise": _FP8_BLOCKWISE_CONFIGS.copy(),
    }


def update_config_gfx950(
    dtype_str: str,
    M: int,
    config: Dict[str, Any],
) -> None:
    """Update a configuration entry (used by autotuning)."""
    if dtype_str == "fp8_w8a8":
        _FP8_CONFIGS[M] = config
    elif dtype_str == "fp4_w4a8":
        _FP4_CONFIGS[M] = config
    elif dtype_str == "fp8_blockwise":
        _FP8_BLOCKWISE_CONFIGS[M] = config
    else:
        _BF16_CONFIGS[M] = config
