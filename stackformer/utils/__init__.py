"""General, device, attention, and feed-forward utility helper functions.

Exposes:
    - get_device, print_device_info, clear_cuda_cache, synchronize, format_memory, move_to_device: Device utilities
    - count_parameters, ensure_dir, get_rank, get_world_size, is_main_process, print_once, seed_everything, timestamp: General utilities
    - hidden_dim_helper: Feed-forward dimension calculator
    - _run_sdpa, _normalize_mask_type, _get_attention_mask: Attention helper functions
"""

from .attn_utils import _get_attention_mask, _normalize_mask_type, _run_sdpa
from .cache import _grad_safe_splice
from .device import clear_cuda_cache, format_memory, get_device, move_to_device, print_device_info, synchronize
from .ff_utils import hidden_dim_helper
from .utils import (
    count_parameters,
    ensure_dir,
    get_rank,
    get_world_size,
    is_main_process,
    print_once,
    seed_everything,
    timestamp,
)

__all__ = [
    "hidden_dim_helper",
    "get_device",
    "print_device_info",
    "clear_cuda_cache",
    "synchronize",
    "format_memory",
    "seed_everything",
    "move_to_device",
    "ensure_dir",
    "count_parameters",
    "timestamp",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "print_once",
    "_run_sdpa",
    "_normalize_mask_type",
    "_get_attention_mask",
    "_grad_safe_splice",
]

