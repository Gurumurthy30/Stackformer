"""Attention mask creation utilities for FlexAttention block-sparse layouts.

Companion to ``attention_masks_dense.py``: same mask names, same parameters,
same composition semantics (``combine="or"``/``"and"``), but built on top of
``torch.nn.attention.flex_attention`` so masking is expressed as compiled
block-sparsity (``BlockMask``) instead of a dense ``(seq_len, seq_len)`` boolean
tensor. Use this version when sequences are long enough that materializing a
dense mask would be wasteful.

Core FlexAttention concept, ``mask_mod``:
    A function with signature ``(b, h, q_idx, kv_idx) -> Tensor[bool]`` where
    ``b``/``h``/``q_idx``/``kv_idx`` are traced index tensors (batch, head,
    query position, key position). It follows the same True=visible /
    False=masked convention as SDPA. ``create_block_mask`` compiles a
    ``mask_mod`` into a ``BlockMask`` that FlexAttention's kernel uses to skip
    fully-masked blocks entirely.

Unlike the dense functions, mask_mod closures can't take arbitrary keyword
arguments at call time (FlexAttention always calls them as
``mask_mod(b, h, q_idx, kv_idx)``), so every parameterized mask below is a
*factory*: a function that takes your parameters (``window_size``,
``global_index``, ...) up front and returns the actual ``mask_mod`` closure.
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Literal, Optional

import torch
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    noop_mask,
    or_masks,
)

MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

def causal(b, h, q_idx, kv_idx):
    """Standard autoregressive causal ``mask_mod``: key position <= query position."""
    return q_idx >= kv_idx


def sliding_window_mask_mod(window_size: int) -> MaskMod:
    """Factory for a sliding-window causal ``mask_mod``.

    Args:
        window_size (int): Number of past positions (inclusive of the query
            itself) each query may attend to.

    Returns:
        MaskMod: ``mask_mod`` closure capturing ``window_size``.
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    def sliding_window(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)

    return sliding_window


def dilated_mask_mod(dilation: int) -> MaskMod:
    """Factory for a dilated-stride ``mask_mod`` (not restricted to the past).

    Args:
        dilation (int): Keep only key positions whose distance from the query
            is a multiple of ``dilation``.

    Returns:
        MaskMod: ``mask_mod`` closure capturing ``dilation``.
    """
    if dilation <= 0:
        raise ValueError("dilation must be > 0")

    def dilated(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx) % dilation == 0

    return dilated


def dilated_causal_mask_mod(dilation: int) -> MaskMod:
    """Factory combining ``causal`` and ``dilated_mask_mod`` (AND).

    Args:
        dilation (int): Dilation stride factor.

    Returns:
        MaskMod: ``mask_mod`` closure for a causal, dilated-stride pattern.
    """
    return and_masks(causal, dilated_mask_mod(dilation))


def global_mask_mod(
    seq_len: int, global_index: List[int]) -> MaskMod:
    """Factory for a global-attention ``mask_mod``.

    Any query or key at one of ``global_index`` positions is globally
    visible: it can attend to everything, and everything can attend to it
    (matches ``global_mask`` semantics in ``attention_masks_dense.py``).

    Args:
        seq_len (int): Sequence length ``T``, used to size and validate the
            lookup tensor.
        global_index (List[int]): Token positions with global visibility.

    Returns:
        MaskMod: ``mask_mod`` closure capturing a boolean ``(seq_len,)``
            lookup tensor.
    """
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
    """Factory for a random sparse causal ``mask_mod``.

    Precomputes, for every query row, ``num_random`` random past (causal) key
    positions to attend to, then exposes the lookup as a ``mask_mod`` closure.
    (Note: precomputing an ``(seq_len, seq_len)`` pattern tensor defeats some
    of the memory benefit of block-sparsity for very long sequences; it's fine
    for moderate lengths, matching the dense version's semantics exactly.)

    Args:
        seq_len (int): Sequence length ``T`` (needed up front to precompute the pattern).
        num_random (int): Number of random past positions per query.
        generator (torch.Generator, optional): RNG generator for reproducible sampling.

    Returns:
        MaskMod: ``mask_mod`` closure capturing a precomputed boolean pattern tensor.
    """
    if num_random < 0 or num_random > seq_len:
        raise ValueError("num_random must be between 0 and seq_len")

    scores = torch.rand(seq_len, seq_len, device="cuda", generator=generator).tril()
    cols = scores.topk(num_random, dim=1).indices
    pattern = torch.zeros(seq_len, seq_len, dtype=torch.bool, device="cuda")
    pattern.scatter_(1, cols, True)

    def random_fn(b, h, q_idx, kv_idx):
        return pattern[q_idx, kv_idx]

    return random_fn


def document_mask_mod(
    seq_len: int, document_ids: torch.Tensor) -> MaskMod:
    """Factory for a document (packed-sequence) ``mask_mod``.

    Used when multiple documents are packed end-to-end into one sequence; a
    token may only attend to other tokens from the same document. Typically
    combined with ``causal`` via ``and_masks`` / ``make_block_mask(["causal",
    "document_mask"], combine="and", ...)``.

    Args:
        seq_len (int): Sequence length ``T``, used to validate ``document_ids``.
        document_ids (torch.Tensor): 1D integer tensor of shape ``(seq_len,)``
            giving the document id each token belongs to.

    Returns:
        MaskMod: ``mask_mod`` closure that is True iff query and key share a document id.
    """
    if document_ids.dim() != 1 or document_ids.shape[0] != seq_len:
        raise ValueError(
            f"document_ids must be a 1D tensor of length seq_len={seq_len}, "
            f"got shape {tuple(document_ids.shape)}"
        )

    def document_fn(b, h, q_idx, kv_idx):
        return document_ids[q_idx] == document_ids[kv_idx]

    return document_fn


def mistral_mask_mod(window_size: int, dilation: int) -> MaskMod:
    """Factory combining sliding-window and dilated-causal ``mask_mod``s (OR).

    Args:
        window_size (int): Local window size.
        dilation (int): Dilation stride.

    Returns:
        MaskMod: ``mask_mod`` closure for the hybrid Mistral-style pattern.
    """
    return or_masks(dilated_causal_mask_mod(dilation), sliding_window_mask_mod(window_size))

# Registory

def _no_mask_builder(seq_len: int, **kwargs) -> MaskMod:
    return noop_mask


def _causal_builder(seq_len: int, **kwargs) -> MaskMod:
    return causal


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
    """Construct a compiled FlexAttention ``BlockMask`` from named mask strategies.

    Mirrors ``make_mask`` in ``attention_masks_dense.py``: pass one or more
    mask names and any parameters they need, and get back a ready-to-use
    ``BlockMask`` for ``flex_attention``.

    Args:
        mask_types (list[str] | tuple[str, ...] | str | None): List, tuple, string,
            or None of mask names. ``None`` is treated as ``"no"`` (full attention).
        B (int): Batch size (or ``1`` to broadcast across the batch).
        H (int): Number of heads (or ``1`` to broadcast across heads).
        Q_LEN (int): Query sequence length.
        KV_LEN (int): Key/value sequence length.
        combine (Literal["or", "and"], default="or"): How to combine multiple
            mask types: ``or_masks`` / ``and_masks`` under the hood.
        **kwargs: Additional parameters passed to the relevant mask factories
            (e.g. ``window_size``, ``dilation``, ``num_random``, ``generator``,
            ``global_index``, ``document_ids``). Each factory only receives the
            kwargs matching its own signature, so it's safe to pass the union
            of every parameter you need across all requested mask types.

    Returns:
        BlockMask: Compiled block-sparse mask for use with
            ``torch.nn.attention.flex_attention.flex_attention``.

    Example:
        >>> # Causal attention restricted to within-document spans (packed sequences)
        >>> make_block_mask(
        ...     ["causal", "document_mask"],
        ...     B=1, H=1, Q_LEN=8, KV_LEN=8,
        ...     combine="and",
        ...     document_ids=torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]),
        ... )
        >>> # Global tokens 0 and 5 OR'd with a local sliding window
        >>> make_block_mask(
        ...     ["global_mask", "sliding_window"],
        ...     B=1, H=1, Q_LEN=1024, KV_LEN=1024,
        ...     global_index=[0, 5], window_size=128,
        ... )
    """
    if Q_LEN != KV_LEN:
        # Every mask_mod above assumes a shared query/key index space
        # (e.g. causal's q_idx >= kv_idx). Cross-attention-style masks with
        # Q_LEN != KV_LEN would need different mask_mod definitions.
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