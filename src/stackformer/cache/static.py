"""Static key-value cache implementation for autoregressive inference acceleration.

Provides `StaticKVCache`, which allocates fixed contiguous memory buffers ``(B, H, Lmax, D)``
to avoid dynamic memory reallocations during sequence decoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from stackformer.utils.cache import _grad_safe_splice


class StaticKVCache(nn.Module):
    """Pre-allocated contiguous buffer KV cache strategy.

    Simple explanation:
        `StaticKVCache` reserves contiguous memory buffers ``(B, H, Lmax, D)`` upon initialization,
        allowing sequence generation to update key and value slices in-place without dynamic reallocations.

    Constructor args:
        None (initialized via ``allocate()`` method).

    Learnable parameters:
        None. Registered persistent=False buffer tensors ``cache_keys`` and ``cache_values``.

    Forward args:
        None (uses ``update()`` and ``get_kv()`` interface).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Key and Value cache tensors of shape ``(B, H, S, D)``.
    """

    def allocate(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device | str,
        dtype: torch.dtype,
        **kw: object,
    ) -> None:
        """Allocate fixed key and value buffer tensors.

        Args:
            batch_size (int): Maximum batch size (B).
            num_heads (int): Number of attention heads (H).
            head_dim (int): Per-head feature dimension (D).
            max_len (int): Maximum sequence context length (Lmax).
            device (torch.device | str): Target compute device.
            dtype (torch.dtype): Tensor data type.
            **kw (object): Unused keyword arguments.
        """
        self.max_len = max_len
        self.register_buffer(
            "cache_keys",
            torch.empty(batch_size, num_heads, max_len, head_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "cache_values",
            torch.empty(batch_size, num_heads, max_len, head_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self._live: tuple[torch.Tensor, torch.Tensor, int, int] | None = None
        self._B: int = batch_size

    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        """Update static KV cache buffers with new key and value tensors.

        Args:
            k (torch.Tensor): Key projection tensor of shape ``(B, H, T, D)``.
            v (torch.Tensor): Value projection tensor of shape ``(B, H, T, D)``.
            start_pos (int): Starting sequence index.
        """
        B, H, T, D = k.shape
        end_pos = start_pos + T
        assert end_pos <= self.max_len, (
            f"KV cache capacity exceeded: start_pos={start_pos} + T={T} = {end_pos} "
            f"> cache length {self.max_len}"
        )
        assert k.dtype == self.cache_keys.dtype, (
            f"KV cache dtype mismatch: got {k.dtype}, cache is {self.cache_keys.dtype}. "
            f"Cast inputs or recreate the cache with the desired dtype."
        )

        k_store = k.detach() if not torch.is_grad_enabled() else k
        v_store = v.detach() if not torch.is_grad_enabled() else v

        self.cache_keys[:B, :, start_pos:end_pos] = k_store
        self.cache_values[:B, :, start_pos:end_pos] = v_store

        self._B = B
        self._live = (k, v, start_pos, end_pos)

    def get_kv(self, start_pos: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve key and value tensors up to end_pos, preserving autograd graphs.

        Args:
            start_pos (int): Start sequence position index.
            end_pos (int): End sequence position index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Key and Value tensors of shape ``(B, H, S, D)``.
        """
        assert self._live is not None, "update() must be called before get_kv()"
        k_live, v_live, sp, ep = self._live
        assert ep == end_pos, (
            "get_kv() span doesn't match the most recent update() call — "
            "call update() then get_kv() with the same (start_pos, end_pos) each step."
        )

        k_full = self.cache_keys[: self._B, :, :end_pos]
        v_full = self.cache_values[: self._B, :, :end_pos]

        k_full = _grad_safe_splice(k_full, k_live, sp, ep)
        v_full = _grad_safe_splice(v_full, v_live, sp, ep)
        return k_full, v_full

    def peek_kv(
        self, start_pos: int = 0, end_pos: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read-only inspection of key and value buffers without autograd node splicing.

        Args:
            start_pos (int, default=0): Start sequence position index.
            end_pos (int | None, default=None): Optional end position index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Key and Value buffer slices.
        """
        if end_pos is None:
            end_pos = self.max_len
        return (
            self.cache_keys[: self._B, :, start_pos:end_pos],
            self.cache_values[: self._B, :, start_pos:end_pos],
        )

    def reset(self) -> None:
        """Reset key and value buffers to zero and clear live state pointers."""
        self.cache_keys.zero_()
        self.cache_values.zero_()
        self._live = None