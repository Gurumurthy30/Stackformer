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

from stackformer.modules.masks import (
    MaskSpec,
    _get_attention_mask,
    _get_block_mask,
    make_block_mask,
    make_mask,
)

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
    backend: Backend = "sdpa",
    q: torch.Tensor = None,
    k: torch.Tensor = None,
    v: torch.Tensor = None,
    mask_type: MaskSpec = None,
    seq_len: Optional[int] = None,
    device: torch.device | str | None = None,
    cache: Optional[OrderedDict] = None,
    dropout_p: float = 0.0,
    combine: Literal["or", "and"] = "or",
    score_mod: Optional[Callable] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    mask_kwargs: Optional[Dict[str, Any]] = None,
    attn_mask: Any = None,
) -> torch.Tensor:
    """Dispatch attention computation to SDPA or FlexAttention backend based on `backend`.

    Args:
        backend (Backend, default="sdpa"): Attention backend ("sdpa" or "flex").
        q (torch.Tensor): Query tensor of shape `(B, H, T, D)`.
        k (torch.Tensor): Key tensor of shape `(B, H_kv, S, D)`.
        v (torch.Tensor): Value tensor of shape `(B, H_kv, S, D)`.
        mask_type (MaskSpec, optional): Mask specification ('causal', 'sliding_window', etc.).
        seq_len (int, optional): Sequence length. Defaults to `q.size(-2)`.
        device (torch.device | str, optional): Device for mask creation. Defaults to `q.device`.
        cache (OrderedDict, optional): LRU cache for mask storage.
        dropout_p (float, default=0.0): Attention dropout probability (sdpa only).
        combine (Literal["or", "and"], default="or"): How multiple masks combine.
        score_mod (Callable, optional): Score modifier callback for FlexAttention.
        scale (float, optional): Custom scale factor for QK^T.
        enable_gqa (bool, default=False): Enable GQA broadcasting in attention kernels.
        mask_kwargs (Dict[str, Any], optional): Additional kwargs for mask builder.
        attn_mask (Any, optional): Explicit pre-computed attention mask (dense tensor or BlockMask).

    Returns:
        torch.Tensor: Output tensor of shape `(B, H, T, D)`.
    """
    if backend not in ("sdpa", "flex"):
        raise ValueError(f"Unknown backend '{backend}'. Expected 'sdpa' or 'flex'.")

    mask_kwargs = mask_kwargs or {}
    if seq_len is None and q is not None:
        seq_len = q.size(-2)
    if device is None and q is not None:
        device = q.device

    if backend == "sdpa":
        if attn_mask is None and mask_type is not None:
            if cache is None:
                attn_mask = make_mask(mask_type, seq_len=seq_len, device=device, combine=combine, **mask_kwargs)
            else:
                attn_mask = _get_attention_mask(
                    cache, mask_type, seq_len, device, combine=combine, **mask_kwargs
                )
        return _run_sdpa(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale, enable_gqa=enable_gqa
        )

    if backend == "flex":
        block_mask = attn_mask
        if block_mask is None and mask_type is not None:
            if cache is None:
                block_mask = make_block_mask(
                    mask_type, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=device, combine=combine, **mask_kwargs
                )
            else:
                block_mask = _get_block_mask(
                    cache, mask_type, seq_len, device, combine=combine, **mask_kwargs
                )
        return _run_flex_attention(
            q, k, v, score_mod=score_mod, block_mask=block_mask, scale=scale, enable_gqa=enable_gqa
        )

