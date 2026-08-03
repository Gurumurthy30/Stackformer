"""Attention mask creation utilities for dense and sparse Transformer layouts.

Follows PyTorch Scaled Dot Product Attention (SDPA) boolean mask conventions:
    - True: position is visible (unmasked)
    - False: position is masked out (-inf added to attention logits)

Mask dimensions:
    Shape ``(seq_len, seq_len)`` where row=query index, col=key index.
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Literal, Optional

import torch

def causal(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Standard autoregressive causal attention mask.

    Args:
        seq_len (int): Sequence length ``T``.
        device (torch.device | str | None, default=None): Compute device.

    Returns:
        torch.Tensor: Boolean lower-triangular mask tensor of shape ``(seq_len, seq_len)``.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def sliding_window(
    seq_len: int, window_size: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Sliding window causal attention mask.

    Args:
        seq_len (int): Sequence length ``T``.
        window_size (int): Size of local attention window.
        device (torch.device | str | None, default=None): Compute device.

    Returns:
        torch.Tensor: Boolean sliding window mask tensor of shape ``(seq_len, seq_len)``.
    """
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
    """Dilated causal attention mask.

    Args:
        seq_len (int): Sequence length ``T``.
        dilation (int): Dilation stride factor.
        device (torch.device | str | None, default=None): Compute device.

    Returns:
        torch.Tensor: Boolean dilated mask tensor of shape ``(seq_len, seq_len)``.
    """
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
    """Random sparse causal attention mask.

    For every query row, ``num_random`` past (causal) key positions are
    sampled uniformly at random and marked visible.

    Args:
        seq_len (int): Sequence length ``T``.
        num_random (int): Number of random past positions to attend to.
        device (torch.device | str | None, default=None): Compute device.
        generator (torch.Generator, optional): RNG generator for reproducible sampling.

    Returns:
        torch.Tensor: Boolean random sparse mask tensor of shape ``(seq_len, seq_len)``.
    """
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
    """Global attention mask.

    Tokens listed in ``global_index`` are made globally visible: they can
    attend to every position, and every position can attend to them (their
    entire row and column are set to True). This is the same semantics used
    by the FlexAttention version of this mask in ``attention_masks_flex.py``,
    so the two are interchangeable.

    Args:
        seq_len (int): Sequence length ``T``.
        global_index (List[int]): Indices of tokens with global visibility.
        device (torch.device | str | None, default=None): Compute device.

    Returns:
        torch.Tensor: Boolean global attention mask tensor of shape ``(seq_len, seq_len)``.
    """
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
    """Document (packed-sequence) attention mask.

    Used when multiple independent documents are packed end-to-end into a
    single training sequence (e.g. for efficient batching). A token may only
    attend to other tokens that belong to the same document. Typically
    combined with ``causal`` (via ``make_mask(["causal", "document_mask"],
    combine="and")``) to get a causal-within-document mask.

    Args:
        seq_len (int): Sequence length ``T``.
        document_ids (torch.Tensor): 1D integer tensor of shape ``(seq_len,)``
            giving the document id each token belongs to (e.g.
            ``[0, 0, 0, 1, 1, 2, 2, 2]`` for 3 packed documents).
        device (torch.device | str | None, default=None): Compute device.

    Returns:
        torch.Tensor: Boolean mask tensor of shape ``(seq_len, seq_len)`` that
            is True where the query and key positions share a document id.
    """
    if document_ids.dim() != 1 or document_ids.shape[0] != seq_len:
        raise ValueError(
            f"document_ids must be a 1D tensor of length seq_len={seq_len}, "
            f"got shape {tuple(document_ids.shape)}"
        )

    document_ids = document_ids.to(device=device)
    return document_ids.unsqueeze(1) == document_ids.unsqueeze(0)


def no_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Full unmasked bidirectional attention mask.

    Args:
        seq_len (int): Sequence length ``T``.
        device (torch.device | str | None, default=None): Compute device.

    Returns:
        torch.Tensor: Boolean all-True mask tensor of shape ``(seq_len, seq_len)``.
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)


def mistral(
    seq_len: int, window_size: int, dilation: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """Mistral-style hybrid sliding window + dilated attention mask.

    Args:
        seq_len (int): Sequence length ``T``.
        window_size (int): Local window size.
        dilation (int): Dilation stride.
        device (torch.device | str | None, default=None): Compute device.

    Returns:
        torch.Tensor: Hybrid boolean mask tensor of shape ``(seq_len, seq_len)``.
    """
    return sliding_window(seq_len, window_size, device=device) | dilated_causal(
        seq_len, dilation, device=device
    )

MASK_REGISTRY: Dict[str, Callable] = {
    "no": no_mask,
    "causal": causal,
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
    """Construct a composite boolean attention mask from specified mask strategy names.

    Args:
        mask_types (list[str] | tuple[str, ...] | str | None): List, tuple, string, or None of mask names.
        seq_len (int): Sequence length ``T``.
        device (torch.device | str | None, default=None): Target compute device.
        combine (Literal["or", "and"], default="or"): Operator to combine multiple masks.
        **kwargs: Additional parameters passed to mask builder functions (e.g. ``window_size``,
            ``dilation``, ``num_random``, ``global_index``, ``document_ids``). Each builder
            function only receives the kwargs matching its own signature, so it's safe to
            pass the union of every parameter you need across all requested mask types.

    Returns:
        torch.Tensor: Composite boolean mask tensor of shape ``(seq_len, seq_len)``.

    Example:
        >>> # Causal attention restricted to within-document spans (packed sequences)
        >>> make_mask(
        ...     ["causal", "document_mask"],
        ...     seq_len=8,
        ...     combine="and",
        ...     document_ids=torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]),
        ... )
    """
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