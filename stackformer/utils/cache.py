"""Gradient-safe KV cache tensor splicing utilities for autograd execution.

Provides helper function `_grad_safe_splice` to preserve autograd computation graphs when reading from
detached persistent memory buffers during training.
"""

from __future__ import annotations

import torch


def _grad_safe_splice(
    cached: torch.Tensor, live: torch.Tensor, start_pos: int, end_pos: int
) -> torch.Tensor:
    """Splice live autograd-tracked tensors over detached historical buffer slices.

    When autograd is enabled, in-place writes into persistent buffers corrupt computation graphs.
    This helper clones the historical prefix and splices the live (grad-tracked) chunk back in.

    Args:
        cached (torch.Tensor): Detached historical buffer tensor of shape ``(B, H, S, D)``.
        live (torch.Tensor): Live grad-tracked tensor of shape ``(B, H, T, D)``.
        start_pos (int): Sequence start index.
        end_pos (int): Sequence end index.

    Returns:
        torch.Tensor: Spliced tensor preserving autograd connections for live tokens.
    """
    if not torch.is_grad_enabled():
        return cached
    out = cached.detach().clone()
    out[:, :, start_pos:end_pos] = live
    return out