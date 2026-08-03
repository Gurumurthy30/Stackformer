"""Attention module execution and mask caching utilities.

Provides scaled dot-product attention (SDPA) wrapper, mask type normalizer, and mask caching helper.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, List

import torch
from stackformer.attention_engine.masking.dense import make_mask
from stackformer.attention_engine.masking.mask_cache import _get_block_mask
# Maximum number of cached attention masks.
_MAX_MASK_CACHE_SIZE = 32


def _normalize_mask_type(mask_type: bool | str | list[str] | tuple[str, ...] | None) -> List[str] | None:
    """Normalize user-provided mask_type into a canonical list of string identifiers.

    Args:
        mask_type (bool | str | list[str] | tuple[str, ...] | None): Mask type input.

    Returns:
        List[str] | None: Canonical list of mask string names, or None.
    """
    if mask_type is True:
        return ["causal"]

    if mask_type in (False, None):
        return None

    if isinstance(mask_type, str):
        return [mask_type]

    if isinstance(mask_type, (list, tuple)):
        return list(mask_type)

    raise TypeError("mask_type must be bool, str, or list of str")


def _canonical_device(device: str | torch.device) -> torch.device:
    """Convert a device specification into a canonical `torch.device`.

    Ensures cache keys remain consistent between 'cuda' and 'cuda:0'.

    Args:
        device (str | torch.device): Input compute device.

    Returns:
        torch.device: Canonicalized torch.device instance.
    """
    dev = torch.device(device)

    if dev.type == "cuda" and dev.index is None:
        current = torch.cuda.current_device() if torch.cuda.is_available() else 0
        dev = torch.device(f"cuda:{current}")

    return dev


def _get_attention_mask(
    cache: OrderedDict,
    mask_type: Any,
    seq_len: int,
    device: str | torch.device,
    **mask_kwargs: Any,
) -> torch.Tensor | None:
    """Retrieve or create a cached attention mask using FIFO eviction.

    Args:
        cache (OrderedDict): Attention mask cache dictionary.
        mask_type (Any): Mask specification identifier.
        seq_len (int): Target sequence length.
        device (str | torch.device): Target compute device.
        **mask_kwargs (Any): Additional arguments forwarded to `make_mask()`.

    Returns:
        torch.Tensor | None: Constructed attention mask tensor or None.
    """

    mask_types = _normalize_mask_type(mask_type)

    if mask_types is None:
        return None

    canonical_dev = _canonical_device(device)

    key = (
        seq_len,
        tuple(mask_types),
        canonical_dev,
    )

    if key not in cache:
        if len(cache) >= _MAX_MASK_CACHE_SIZE:
            cache.popitem(last=False)

        cache[key] = make_mask(
            mask_types,
            seq_len,
            device=canonical_dev,
            **mask_kwargs,
        )

    return cache[key]

