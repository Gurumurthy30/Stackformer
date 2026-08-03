"""Attention module execution and mask caching utilities.

Provides scaled dot-product attention (SDPA) wrapper, mask type normalizer, and mask caching helper.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, List

import torch
from stackformer.modules.masks import (
    _get_attention_mask,
    _get_block_mask,
    make_mask,
)

# Maximum number of cached attention masks.
_MAX_MASK_CACHE_SIZE = 32


def _normalize_mask_type(mask_type: bool | str | list[str] | tuple[str, ...] | None) -> List[str] | None:
    """Normalize user-provided mask_type into a canonical list of string identifiers."""
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
    """Convert a device specification into a canonical `torch.device`."""
    dev = torch.device(device)

    if dev.type == "cuda" and dev.index is None:
        current = torch.cuda.current_device() if torch.cuda.is_available() else 0
        dev = torch.device(f"cuda:{current}")

    return dev


