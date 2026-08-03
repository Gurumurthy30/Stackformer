"""Caches for turning named mask strategies (from ``attention_masks_dense.py``
/ ``attention_masks_flex.py``) into ready-to-use mask objects, keyed so
identical requests (same mask names, seq_len, device, and static kwargs)
don't get rebuilt every forward pass.

This is the glue between "which mask(s) do I want" (the two mask-registry
files) and "which attention kernel am I running" (``attention_backends.py``):
    - SDPA wants a dense ``(T, T)`` boolean ``attn_mask`` -> ``_get_attention_mask``.
    - FlexAttention wants a compiled ``BlockMask`` -> ``_get_block_mask``.

Caching notes:
    - Cache keys only cover *hashable, static* kwargs (ints, floats, strings,
      bools, tuples/lists of those). Any kwarg that's a ``torch.Tensor`` or
      ``torch.Generator`` (e.g. ``document_ids``, a random-mask ``generator``)
      makes the request uncacheable, since such a mask is expected to vary
      call-to-call (different packed-sequence layout, fresh random pattern).
      Those masks are still built correctly -- they just aren't memoized.
    - Both caches are simple size-capped LRUs (``OrderedDict`` + evict-oldest).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch.nn.attention.flex_attention import BlockMask

from .dense import make_mask
from .functional import make_block_mask

_DEFAULT_MAX_CACHE_SIZE = 32

MaskSpec = Union[List[str], Tuple[str, ...], str, None]


def _normalize_mask_names(mask_type: MaskSpec) -> Tuple[str, ...]:
    """Normalize a mask spec into a sorted tuple of lowercase names.

    Sorting is safe because both ``combine="or"`` and ``combine="and"`` are
    commutative and associative, so mask order never changes the result --
    it only changes cache-key stability, which sorting fixes.
    """
    if mask_type is None:
        return ("no",)
    if isinstance(mask_type, str):
        return (mask_type.lower(),)
    return tuple(sorted(name.lower() for name in mask_type))


def _make_cache_key(
    mask_type: MaskSpec,
    seq_len: int,
    device: torch.device | str | None,
    combine: str,
    mask_kwargs: Dict[str, Any],
) -> Optional[Tuple]:
    """Build a hashable cache key, or return None if the request can't be cached.

    Args:
        mask_type (MaskSpec): Mask name(s) as accepted by ``make_mask`` / ``make_block_mask``.
        seq_len (int): Sequence length.
        device (torch.device | str | None): Target device.
        combine (str): ``"or"`` or ``"and"``.
        mask_kwargs (Dict[str, Any]): Extra kwargs passed to the mask builder(s).

    Returns:
        Optional[Tuple]: A hashable key, or None if ``mask_kwargs`` contains a
            tensor/generator (i.e. the mask is inherently call-specific).
    """
    names = _normalize_mask_names(mask_type)

    hashable_kwargs = []
    for key in sorted(mask_kwargs):
        value = mask_kwargs[key]
        if isinstance(value, (torch.Tensor, torch.Generator)):
            return None
        if isinstance(value, list):
            value = tuple(value)
        hashable_kwargs.append((key, value))

    return (names, seq_len, str(device), combine, tuple(hashable_kwargs))


def _cache_put(cache: "OrderedDict", key, value, max_size: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _get_attention_mask(
    cache: "OrderedDict",
    mask_type: MaskSpec,
    seq_len: int,
    device: torch.device | str | None,
    combine: Literal["or", "and"] = "or",
    max_cache_size: int = _DEFAULT_MAX_CACHE_SIZE,
    **mask_kwargs,
) -> Optional[torch.Tensor]:
    """Get (or build + cache) a dense boolean attention mask for SDPA.

    Args:
        cache (OrderedDict): Per-module cache dict, e.g. ``self._mask_cache``.
        mask_type (MaskSpec): Mask name(s), e.g. ``"causal"`` or
            ``["causal", "document_mask"]``. ``None`` returns ``None``
            (full attention, no mask needed).
        seq_len (int): Sequence length ``T``.
        device (torch.device | str | None): Target device.
        combine (Literal["or", "and"], default="or"): How to combine multiple mask types.
        max_cache_size (int, default=32): Max number of distinct masks to retain.
        **mask_kwargs: Forwarded to ``make_mask`` (e.g. ``window_size``,
            ``dilation``, ``global_index``, ``document_ids``, ``num_random``).

    Returns:
        Optional[torch.Tensor]: Boolean mask of shape ``(seq_len, seq_len)``,
            or None if ``mask_type`` is None.
    """
    if mask_type is None:
        return None

    key = _make_cache_key(mask_type, seq_len, device, combine, mask_kwargs)
    if key is not None and key in cache:
        cache.move_to_end(key)
        return cache[key]

    mask = make_mask(mask_type, seq_len=seq_len, device=device, combine=combine, **mask_kwargs)

    if key is not None:
        _cache_put(cache, key, mask, max_cache_size)

    return mask


def _get_block_mask(
    cache: "OrderedDict",
    mask_type: MaskSpec,
    seq_len: int,
    device: torch.device | str | None,
    combine: Literal["or", "and"] = "or",
    max_cache_size: int = _DEFAULT_MAX_CACHE_SIZE,
    **mask_kwargs,
) -> Optional[BlockMask]:
    """Get (or build + cache) a compiled ``BlockMask`` for FlexAttention.

    Always builds the block mask with ``B=1, H=1``: FlexAttention broadcasts
    a ``BlockMask`` created this way across any real batch size / head count,
    which keeps the cache small (one entry per mask config, independent of
    batch size) and matches the common FlexAttention usage pattern.

    Args:
        cache (OrderedDict): Per-module cache dict, e.g. ``self._mask_cache``.
        mask_type (MaskSpec): Mask name(s). ``None`` returns ``None`` (full attention).
        seq_len (int): Sequence length (used for both ``Q_LEN`` and ``KV_LEN``;
            self-attention only -- see ``make_block_mask``).
        device (torch.device | str | None): Target device.
        combine (Literal["or", "and"], default="or"): How to combine multiple mask types.
        max_cache_size (int, default=32): Max number of distinct block masks to retain.
        **mask_kwargs: Forwarded to ``make_block_mask`` (e.g. ``window_size``,
            ``dilation``, ``global_index``, ``document_ids``, ``num_random``).

    Returns:
        Optional[BlockMask]: Compiled block mask, or None if ``mask_type`` is None.
    """
    if mask_type is None:
        return None

    key = _make_cache_key(mask_type, seq_len, device, combine, mask_kwargs)
    if key is not None and key in cache:
        cache.move_to_end(key)
        return cache[key]

    block_mask = make_block_mask(
        mask_type,
        B=1,
        H=1,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        combine=combine,
        **mask_kwargs,
    )

    if key is not None:
        _cache_put(cache, key, block_mask, max_cache_size)

    return block_mask