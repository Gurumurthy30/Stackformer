"""Attention execution engine backends and dispatcher for StackFormer.

Provides wrappers around PyTorch SDPA and FlexAttention, along with the high-level
`_run_attention` backend dispatcher function.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from stackformer.modules.masks import make_block_mask, make_mask

Backend = Literal["sdpa", "flex"]


def _run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Wrapper around PyTorch's scaled dot-product attention (SDPA).

    Args:
        q (torch.Tensor): Query tensor of shape `(B, H, T, D)`.
        k (torch.Tensor): Key tensor of shape `(B, H, S, D)`.
        v (torch.Tensor): Value tensor of shape `(B, H, S, D)`.
        attn_mask (torch.Tensor | None): Attention mask tensor or None.
        dropout_p (float, default=0.0): Attention dropout probability.
        is_causal (bool, default=False): Causal flag for SDPA.
        scale (float | None, default=None): Custom scale factor for QK^T.
        enable_gqa (bool, default=False): Enable GQA broadcasting in SDPA.

    Returns:
        torch.Tensor: Output tensor of shape `(B, H, T, D)`.
    """
    kwargs: Dict[str, Any] = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
    }
    if scale is not None:
        kwargs["scale"] = scale
    if enable_gqa:
        kwargs["enable_gqa"] = enable_gqa

    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def _run_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Optional[Callable] = None,
    block_mask: Any = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Wrapper around PyTorch FlexAttention backend."""
    return flex_attention(
        query=query,
        key=key,
        value=value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
    )


def _run_attention(
    backend: Backend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_type: Optional[str],
    seq_len: int,
    device: torch.device | str | None,
    cache: OrderedDict,
    dropout_p: float = 0.0,
    combine: Literal["or", "and"] = "or",
    score_mod: Optional[Callable] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    mask_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Dispatch attention computation to SDPA or FlexAttention backend based on `backend`."""
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
