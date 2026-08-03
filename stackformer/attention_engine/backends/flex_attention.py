import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
    
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def _run_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod = None,
    block_mask = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Optional[Dict[str, Any]] = None):
    
    
    return flex_attention(query = query, key = key, value = value,
                          score_mod = score_mod, block_mask = block_mask, scale = scale,
                          enable_gqa = enable_gqa, return_lse = return_lse, kernel_options = kernel_options)