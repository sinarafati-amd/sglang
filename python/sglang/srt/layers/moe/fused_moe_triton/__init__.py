from contextlib import contextmanager
import os
from typing import Any, Dict, Optional

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    get_config_file_name,
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
    moe_align_block_size,
)

# Import optimized kernel utilities if available
try:
    from sglang.srt.layers.moe.fused_moe_triton.optimized import (
        use_optimized_moe_kernel,
        get_optimized_config,
        get_gpu_arch,
    )
    _optimized_available = True
except ImportError:
    _optimized_available = False
    use_optimized_moe_kernel = lambda: False
    get_optimized_config = lambda *args, **kwargs: None
    get_gpu_arch = lambda: "unknown"


_config: Optional[Dict[str, Any]] = None


def is_optimized_moe_enabled() -> bool:
    """Check if optimized MOE kernel is enabled and available."""
    if not _optimized_available:
        return False
    return use_optimized_moe_kernel()


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> Optional[Dict[str, Any]]:
    return _config


__all__ = [
    "FusedMoE",
    "FusedMoeWeightScaleSupported",
    "override_config",
    "get_config",
    "fused_experts",
    "get_config_file_name",
    "moe_align_block_size",
    "try_get_optimal_moe_config",
    # Optimized kernel exports
    "is_optimized_moe_enabled",
    "use_optimized_moe_kernel",
    "get_optimized_config",
    "get_gpu_arch",
]
