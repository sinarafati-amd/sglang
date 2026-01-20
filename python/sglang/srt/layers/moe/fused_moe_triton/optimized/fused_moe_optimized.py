# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""
Optimized Fused MOE Entry Point for AMD MI300X (gfx942) and MI350 (gfx950).

This module provides the main entry point for the optimized MOE kernels,
handling the full pipeline:
1. Token-expert alignment
2. First GEMM (gate_up projection)
3. Activation (SiLU/GELU) and multiply
4. Second GEMM (down projection)
5. Sum reduction across experts

Usage:
    Set SGLANG_USE_OPTIMIZED_MOE=1 to enable.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_bool_env_var, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import StandardTopKOutput

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

# Import optimized kernels
from .fused_moe_optimized_kernels import (
    act_and_mul_optimized,
    invoke_fused_moe_kernel_optimized_fp4,
    invoke_fused_moe_kernel_optimized_fp8,
    moe_sum_reduce_optimized,
)

# Import baseline utilities for compatibility
from ..moe_align_block_size import moe_align_block_size

if _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul
    if _use_aiter:
        try:
            from aiter import moe_sum
        except ImportError:
            moe_sum = None
    else:
        moe_sum = None


def get_optimized_config_for_shape(
    M: int,
    E: int,
    N: int,
    K: int,
    dtype_str: str,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Get optimized configuration for the given problem shape.
    
    Args:
        M: Number of tokens
        E: Number of experts  
        N: Output dimension
        K: Input dimension
        dtype_str: Data type string (fp8_w8a8, fp4_w4a8, etc.)
        block_shape: Optional block shape for block-wise quantization
    
    Returns:
        Optimized configuration dictionary
    """
    from . import get_gpu_arch
    
    arch = get_gpu_arch()
    
    if arch == "gfx942":
        from .config_gfx942 import get_config_gfx942
        return get_config_gfx942(M, E, N, K, dtype_str, block_shape)
    elif arch == "gfx950":
        from .config_gfx950 import get_config_gfx950
        return get_config_gfx950(M, E, N, K, dtype_str, block_shape)
    else:
        # Fallback to default MFMA-friendly config
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }


@torch.compile(disable=True)  # Disable torch.compile for Triton kernels
def fused_experts_optimized(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    filter_expert: bool = True,
) -> torch.Tensor:
    """
    Optimized fused experts implementation for AMD MI300X/MI350.
    
    This function replaces the baseline fused_experts_impl when
    SGLANG_USE_OPTIMIZED_MOE=1 is set.
    
    Args:
        hidden_states: Input tensor [num_tokens, hidden_size]
        w1: First weight matrix [num_experts, 2*intermediate_size, hidden_size]
        w2: Second weight matrix [num_experts, hidden_size, intermediate_size]
        topk_weights: Expert weights [num_tokens, top_k]
        topk_ids: Expert indices [num_tokens, top_k]
        ... (other args same as baseline)
    
    Returns:
        Output tensor [num_tokens, hidden_size] or [num_tokens, top_k, hidden_size]
    """
    # Validate inputs
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert w1.is_contiguous(), "w1 must be contiguous"
    assert w2.is_contiguous(), "w2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    
    num_tokens, hidden_size = hidden_states.shape
    E, N, _ = w1.shape
    top_k = topk_ids.shape[1]
    
    # Get optimized config
    dtype_str = "fp8_w8a8" if use_fp8_w8a8 else ("fp4_w4a8" if use_fp4_w4a8 else None)
    config = get_optimized_config_for_shape(
        num_tokens, E, N, hidden_size, dtype_str, block_shape
    )
    
    # Align tokens to blocks
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )
    
    # Compute type
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    
    # Allocate intermediate buffers
    total_tokens = sorted_token_ids.shape[0]
    intermediate_cache1 = torch.empty(
        (total_tokens, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    
    # First GEMM: gate_up projection
    if use_fp8_w8a8:
        invoke_fused_moe_kernel_optimized_fp8(
            hidden_states,
            w1,
            b1,
            intermediate_cache1,
            a1_scale,
            w1_scale,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k,
            config,
            compute_type=compute_type,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            c_sorted=False,
            filter_expert=filter_expert,
        )
    elif use_fp4_w4a8:
        invoke_fused_moe_kernel_optimized_fp4(
            hidden_states,
            w1,
            w1_scale,
            b1,
            intermediate_cache1,
            a1_scale,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k,
            config,
            compute_type=compute_type,
            block_shape=block_shape,
            c_sorted=False,
            filter_expert=filter_expert,
        )
    else:
        # BF16/FP16 path - use optimized FP8 kernel with scales=None
        invoke_fused_moe_kernel_optimized_fp8(
            hidden_states,
            w1,
            b1,
            intermediate_cache1,
            None,  # a_scale
            None,  # b_scale
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k,
            {**config, "use_fp8_w8a8": False},
            compute_type=compute_type,
            per_channel_quant=False,
            block_shape=None,
            c_sorted=False,
            filter_expert=filter_expert,
        )
    
    # Activation and multiply
    intermediate_cache2 = torch.empty(
        (total_tokens, N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    
    if activation == "silu":
        if _is_hip and not filter_expert:
            silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        else:
            act_and_mul_optimized(
                intermediate_cache1.view(-1, N),
                intermediate_cache2,
                config,
                topk_ids,
                expert_ids,
                down_moe_use_tma=False,
                activation="silu",
            )
    elif activation == "gelu":
        if _is_hip and not filter_expert:
            gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        else:
            act_and_mul_optimized(
                intermediate_cache1.view(-1, N),
                intermediate_cache2,
                config,
                topk_ids,
                expert_ids,
                down_moe_use_tma=False,
                activation="gelu",
            )
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    # Prepare output buffer
    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, top_k, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)
    
    # Intermediate cache for second GEMM output
    intermediate_cache3 = torch.empty(
        (num_tokens, top_k, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    
    # Second GEMM: down projection
    if use_fp8_w8a8:
        invoke_fused_moe_kernel_optimized_fp8(
            intermediate_cache2,
            w2,
            b2,
            intermediate_cache3 if not no_combine and top_k != 1 else out_hidden_states.unsqueeze(0),
            a2_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            filter_expert=filter_expert,
        )
    elif use_fp4_w4a8:
        invoke_fused_moe_kernel_optimized_fp4(
            intermediate_cache2,
            w2,
            w2_scale,
            b2,
            intermediate_cache3 if not no_combine and top_k != 1 else out_hidden_states.unsqueeze(0),
            a2_scale,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            block_shape=block_shape,
            filter_expert=filter_expert,
        )
    else:
        invoke_fused_moe_kernel_optimized_fp8(
            intermediate_cache2,
            w2,
            b2,
            intermediate_cache3 if not no_combine and top_k != 1 else out_hidden_states.unsqueeze(0),
            None,
            None,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            {**config, "use_fp8_w8a8": False},
            compute_type=compute_type,
            per_channel_quant=False,
            block_shape=None,
            filter_expert=filter_expert,
        )
    
    # Sum reduction
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0
    
    if no_combine:
        pass  # Return [num_tokens, top_k, hidden_size]
    elif top_k == 1 and routed_scaling_factor == 1.0:
        pass  # Already written to out_hidden_states
    elif top_k == 2 and routed_scaling_factor == 1.0:
        torch.add(
            intermediate_cache3[:, 0],
            intermediate_cache3[:, 1],
            out=out_hidden_states,
        )
    else:
        if _use_aiter and moe_sum is not None:
            moe_sum(intermediate_cache3, out_hidden_states)
        else:
            moe_sum_reduce_optimized(
                intermediate_cache3,
                out_hidden_states,
                routed_scaling_factor,
            )
    
    return out_hidden_states


def invoke_fused_moe_kernel_optimized(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output,
    moe_runner_config,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Main entry point for optimized MOE kernel.
    
    This function is called from fused_moe.py when SGLANG_USE_OPTIMIZED_MOE=1.
    """
    topk_weights, topk_ids, _ = topk_output
    
    return fused_experts_optimized(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        b1=b1,
        b2=b2,
        inplace=moe_runner_config.inplace if hasattr(moe_runner_config, 'inplace') else False,
        activation=moe_runner_config.activation if hasattr(moe_runner_config, 'activation') else "silu",
        apply_router_weight_on_input=getattr(moe_runner_config, 'apply_router_weight_on_input', False),
        use_fp8_w8a8=use_fp8_w8a8,
        use_fp4_w4a8=use_fp4_w4a8,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        no_combine=getattr(moe_runner_config, 'no_combine', False),
        routed_scaling_factor=getattr(moe_runner_config, 'routed_scaling_factor', None),
        filter_expert=True,
    )
