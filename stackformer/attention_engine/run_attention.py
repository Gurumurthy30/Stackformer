import torch
import torch.nn.functional as F
from typing import Literal, Optional, Dict, Any
from collections import OrderedDict
 
from .backends.sdpa import _run_sdpa
from .backends.flex_attention import _run_flex_attention
from .masking.dense import make_mask
from .masking.functional import make_block_mask
 
Backend = Literal["sdpa", "flex"]
 
 
def _run_attention(
    backend: Backend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_type: Optional[str],
    seq_len: int,
    device: torch.device | str | None,
    cache: "OrderedDict",
    dropout_p: float = 0.0,
    combine: Literal["or", "and"] = "or",
    score_mod: Optional[callable] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    mask_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    
    mask_kwargs = mask_kwargs or {}
 
    if backend == "sdpa":
        attn_mask = make_mask(
            cache, mask_type, seq_len, device, combine=combine, **mask_kwargs
        )
        return _run_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
 
    if backend == "flex":
        block_mask = make_block_mask(
            cache, mask_type, seq_len, device, combine=combine, **mask_kwargs
        )
        return _run_flex_attention(
            q, k, v, score_mod=score_mod, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa
        )
 
    raise ValueError(f"Unknown backend '{backend}'. Expected 'sdpa' or 'flex'.")
 