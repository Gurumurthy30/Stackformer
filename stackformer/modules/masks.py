"""Attention mask creation, FlexAttention block-mask builders, and mask caching utilities.

Consolidates dense mask builders, FlexAttention functional mask_mods, and LRU cache memoization.
"""

from __future__ import annotations

from collections import OrderedDict
import inspect
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    noop_mask,
    or_masks,
)

MaskSpec = Union[List[str], Tuple[str, ...], str, None]
MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
_DEFAULT_MAX_CACHE_SIZE = 32

# ============================================================================
# Dense Mask Builders (SDPA)
# ============================================================================


def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Standard autoregressive causal attention mask (dense boolean matrix)."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def sliding_window(
    seq_len: int, window_size: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Sliding window causal attention mask."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    i = torch.arange(seq_len, device=device).unsqueeze(1)  # (T, 1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)

    in_past = j <= i
    in_window = (i - j) < window_size

    return in_past & in_window


def dilated_causal(
    seq_len: int, dilation: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Dilated causal attention mask."""
    if dilation <= 0:
        raise ValueError("dilation must be > 0")

    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)

    in_past = j <= i
    on_stride = (i - j) % dilation == 0
    return in_past & on_stride


def random_mask(
    seq_len: int,
    num_random: int,
    device: torch.device | str | None = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Random sparse causal attention mask."""
    if num_random < 0 or num_random > seq_len:
        raise ValueError("num_random must be between 0 and seq_len")
    cols = (
        torch.rand(seq_len, seq_len, device=device, generator=generator)
        .tril()
        .topk(num_random, dim=1)
        .indices
    )

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    mask.scatter_(1, cols, True)
    return mask


def global_mask(
    seq_len: int, global_index: List[int], device: torch.device | str | None = None
) -> torch.Tensor:
    """Global attention mask."""
    if len(global_index) == 0:
        raise ValueError("global_index must contain at least one index")
    if any(i < 0 or i >= seq_len for i in global_index):
        raise ValueError("global_index contains invalid token indices")

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    g = torch.tensor(global_index, device=device)

    mask[g, :] = True
    mask[:, g] = True

    return mask


def document_mask(
    seq_len: int,
    document_ids: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Document (packed-sequence) attention mask."""
    if document_ids.dim() != 1 or document_ids.shape[0] != seq_len:
        raise ValueError(
            f"document_ids must be a 1D tensor of length seq_len={seq_len}, "
            f"got shape {tuple(document_ids.shape)}"
        )

    document_ids = document_ids.to(device=device)
    return document_ids.unsqueeze(1) == document_ids.unsqueeze(0)


def no_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Full unmasked bidirectional attention mask."""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)


def mistral(
    seq_len: int, window_size: int, dilation: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Mistral-style hybrid sliding window + dilated attention mask."""
    return sliding_window(seq_len, window_size, device=device) | dilated_causal(
        seq_len, dilation, device=device
    )


MASK_REGISTRY: Dict[str, Callable] = {
    "no": no_mask,
    "causal": causal_mask,
    "sliding_window": sliding_window,
    "dilated_causal": dilated_causal,
    "random_mask": random_mask,
    "global_mask": global_mask,
    "document_mask": document_mask,
    "mistral": mistral,
}


def make_mask(
    mask_types: list[str] | tuple[str, ...] | str | None,
    seq_len: int,
    device: torch.device | str | None = None,
    combine: Literal["or", "and"] = "or",
    **kwargs,
) -> torch.Tensor:
    """Construct a composite boolean attention mask from specified mask strategy names."""
    if mask_types is None:
        return no_mask(seq_len, device=device)

    if isinstance(mask_types, str):
        mask_types = [mask_types]

    if not isinstance(mask_types, (list, tuple)):
        raise TypeError("mask_types must be a list, tuple, or string of mask name(s)")

    if combine not in ("or", "and"):
        raise ValueError(f"combine must be 'or' or 'and', got {combine!r}")

    init_val = combine == "and"
    mask = torch.full(
        (seq_len, seq_len),
        fill_value=init_val,
        dtype=torch.bool,
        device=device,
    )

    for name in mask_types:
        name = name.lower()
        if name not in MASK_REGISTRY:
            raise ValueError(
                f"Unknown mask '{name}'. Available: {list(MASK_REGISTRY.keys())}"
            )

        fn = MASK_REGISTRY[name]
        sig = inspect.signature(fn)
        call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        partial = fn(seq_len, device=device, **call_kwargs)

        if combine == "or":
            mask |= partial
        else:
            mask &= partial

    return mask


# ============================================================================
# FlexAttention Mask Mods
# ============================================================================


def causal_mask_mod(b, h, q_idx, kv_idx):
    """Standard autoregressive causal mask_mod: key position <= query position."""
    return q_idx >= kv_idx


def sliding_window_mask_mod(window_size: int) -> MaskMod:
    """Factory for a sliding-window causal mask_mod."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    def sliding_window_fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)

    return sliding_window_fn


def dilated_mask_mod(dilation: int) -> MaskMod:
    """Factory for a dilated-stride mask_mod."""
    if dilation <= 0:
        raise ValueError("dilation must be > 0")

    def dilated_fn(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx) % dilation == 0

    return dilated_fn


def dilated_causal_mask_mod(dilation: int) -> MaskMod:
    """Factory combining causal_mask_mod and dilated_mask_mod (AND)."""
    return and_masks(causal_mask_mod, dilated_mask_mod(dilation))


def global_mask_mod(seq_len: int, global_index: List[int]) -> MaskMod:
    """Factory for a global-attention mask_mod."""
    if len(global_index) == 0:
        raise ValueError("global_index must contain at least one index")
    if any(i < 0 or i >= seq_len for i in global_index):
        raise ValueError("global_index contains invalid token indices")

    is_global = torch.zeros(seq_len, dtype=torch.bool, device="cuda")
    is_global[torch.tensor(global_index, device="cuda")] = True

    def global_fn(b, h, q_idx, kv_idx):
        return is_global[q_idx] | is_global[kv_idx]

    return global_fn


def random_mask_mod(
    seq_len: int,
    num_random: int,
    generator: Optional[torch.Generator] = None,
) -> MaskMod:
    """Factory for a random sparse causal mask_mod."""
    if num_random < 0 or num_random > seq_len:
        raise ValueError("num_random must be between 0 and seq_len")

    scores = torch.rand(seq_len, seq_len, device="cuda", generator=generator).tril()
    cols = scores.topk(num_random, dim=1).indices
    pattern = torch.zeros(seq_len, seq_len, dtype=torch.bool, device="cuda")
    pattern.scatter_(1, cols, True)

    def random_fn(b, h, q_idx, kv_idx):
        return pattern[q_idx, kv_idx]

    return random_fn


def document_mask_mod(seq_len: int, document_ids: torch.Tensor) -> MaskMod:
    """Factory for a document (packed-sequence) mask_mod."""
    if document_ids.dim() != 1 or document_ids.shape[0] != seq_len:
        raise ValueError(
            f"document_ids must be a 1D tensor of length seq_len={seq_len}, "
            f"got shape {tuple(document_ids.shape)}"
        )

    def document_fn(b, h, q_idx, kv_idx):
        return document_ids[q_idx] == document_ids[kv_idx]

    return document_fn


def mistral_mask_mod(window_size: int, dilation: int) -> MaskMod:
    """Factory combining sliding-window and dilated-causal mask_mods (OR)."""
    return or_masks(dilated_causal_mask_mod(dilation), sliding_window_mask_mod(window_size))


def _no_mask_builder(seq_len: int, **kwargs) -> MaskMod:
    return noop_mask


def _causal_builder(seq_len: int, **kwargs) -> MaskMod:
    return causal_mask_mod


def _sliding_window_builder(seq_len: int, *, window_size: int, **kwargs) -> MaskMod:
    return sliding_window_mask_mod(window_size)


def _dilated_causal_builder(seq_len: int, *, dilation: int, **kwargs) -> MaskMod:
    return dilated_causal_mask_mod(dilation)


def _random_mask_builder(
    seq_len: int, *, num_random: int, generator: Optional[torch.Generator] = None, **kwargs
) -> MaskMod:
    return random_mask_mod(seq_len, num_random, generator=generator)


def _global_mask_builder(seq_len: int, *, global_index: List[int], **kwargs) -> MaskMod:
    return global_mask_mod(seq_len, global_index)


def _document_mask_builder(seq_len: int, *, document_ids: torch.Tensor, **kwargs) -> MaskMod:
    return document_mask_mod(seq_len, document_ids)


def _mistral_builder(seq_len: int, *, window_size: int, dilation: int, **kwargs) -> MaskMod:
    return mistral_mask_mod(window_size, dilation)


MASK_MOD_REGISTRY: Dict[str, Callable[..., MaskMod]] = {
    "no": _no_mask_builder,
    "causal": _causal_builder,
    "sliding_window": _sliding_window_builder,
    "dilated_causal": _dilated_causal_builder,
    "random_mask": _random_mask_builder,
    "global_mask": _global_mask_builder,
    "document_mask": _document_mask_builder,
    "mistral": _mistral_builder,
}


def make_block_mask(
    mask_types: list[str] | tuple[str, ...] | str | None,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    combine: Literal["or", "and"] = "or",
    **kwargs,
):
    """Construct a compiled FlexAttention BlockMask from named mask strategies."""
    if Q_LEN != KV_LEN:
        raise ValueError(
            "make_block_mask currently assumes self-attention (Q_LEN == KV_LEN); "
            f"got Q_LEN={Q_LEN}, KV_LEN={KV_LEN}"
        )

    if mask_types is None:
        mask_types = "no"

    if isinstance(mask_types, str):
        mask_types = [mask_types]

    if not isinstance(mask_types, (list, tuple)):
        raise TypeError("mask_types must be a list, tuple, or string of mask name(s)")

    if combine not in ("or", "and"):
        raise ValueError(f"combine must be 'or' or 'and', got {combine!r}")

    mask_mods: List[MaskMod] = []
    for name in mask_types:
        name = name.lower()
        if name not in MASK_MOD_REGISTRY:
            raise ValueError(
                f"Unknown mask '{name}'. Available: {list(MASK_MOD_REGISTRY.keys())}"
            )

        builder = MASK_MOD_REGISTRY[name]
        sig = inspect.signature(builder)
        call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        mask_mods.append(builder(Q_LEN, **call_kwargs))

    if len(mask_mods) == 1:
        combined_mod = mask_mods[0]
    else:
        combine_fn = or_masks if combine == "or" else and_masks
        combined_mod = combine_fn(*mask_mods)

    return create_block_mask(combined_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN)


# ============================================================================
# Mask Cache Memoization
# ============================================================================


def _normalize_mask_names(mask_type: MaskSpec) -> Tuple[str, ...]:
    """Normalize a mask spec into a sorted tuple of lowercase names."""
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
    """Build a hashable cache key, or return None if the request can't be cached."""
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


def _cache_put(cache: OrderedDict, key, value, max_size: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _get_attention_mask(
    cache: OrderedDict,
    mask_type: MaskSpec,
    seq_len: int,
    device: torch.device | str | None,
    combine: Literal["or", "and"] = "or",
    max_cache_size: int = _DEFAULT_MAX_CACHE_SIZE,
    **mask_kwargs,
) -> Optional[torch.Tensor]:
    """Get (or build + cache) a dense boolean attention mask for SDPA."""
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
    cache: OrderedDict,
    mask_type: MaskSpec,
    seq_len: int,
    device: torch.device | str | None,
    combine: Literal["or", "and"] = "or",
    max_cache_size: int = _DEFAULT_MAX_CACHE_SIZE,
    **mask_kwargs,
) -> Optional[BlockMask]:
    """Get (or build + cache) a compiled BlockMask for FlexAttention."""
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
