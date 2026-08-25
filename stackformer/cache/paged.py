"""Paged key-value cache implementation for memory-efficient sequence generation.

Provides `PagedKVCache`, which allocates KV cache storage in fixed-size physical blocks from a
shared memory pool rather than contiguous per-sequence allocations.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from stackformer.utils.cache import _grad_safe_splice


class PagedKVCache(nn.Module):
    """Block-pool KV cache module allocating fixed-size physical blocks.

    Simple explanation:
        `PagedKVCache` manages physical key-value memory blocks in a shared pool. Sequences maintain
        a table of block pointers rather than reserving fixed contiguous max-length buffers, reducing
        memory fragmentation during inference.

    Constructor args:
        None (initialized via ``allocate()`` method).

    Learnable parameters:
        None. Registered persistent=False buffer tensors ``pool_keys`` and ``pool_values``.

    Forward args:
        None (uses ``update()`` and ``get_kv()`` interface).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Gathered Key and Value tensors of shape ``(B, H, T, D)``.
    """

    def allocate(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device | str,
        dtype: torch.dtype,
        block_size: int = 16,
        num_blocks: int | None = None,
        **kw: object,
    ) -> None:
        """Allocate physical memory block pools for key and value states.

        Args:
            batch_size (int): Maximum batch size.
            num_heads (int): Number of attention heads (H).
            head_dim (int): Dimensionality per attention head (D).
            max_len (int): Maximum sequence context length.
            device (torch.device | str): Target compute device.
            dtype (torch.dtype): Tensor data type.
            block_size (int, default=16): Number of sequence tokens per block.
            num_blocks (int | None, default=None): Total physical blocks in shared pool.
            **kw (object): Additional unused keyword arguments.
        """
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.batch_size = batch_size

        blocks_per_seq_worst_case = math.ceil(max_len / block_size)
        if num_blocks is None:
            num_blocks = batch_size * blocks_per_seq_worst_case

        self.num_blocks = num_blocks

        self.register_buffer(
            "pool_keys",
            torch.empty(num_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "pool_values",
            torch.empty(num_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype),
            persistent=False,
        )

        self.free_blocks = list(range(num_blocks))
        self.block_tables: list[list[int]] = [[] for _ in range(batch_size)]
        self._live: tuple[torch.Tensor, torch.Tensor, int, int] | None = None
        self._B: int | None = None

    def _alloc_block(self) -> int:
        """Pop a free physical block index from the available pool.

        Returns:
            int: Allocated physical block index.
        """
        assert self.free_blocks, "PagedKVCache: out of physical blocks (pool exhausted)"
        return self.free_blocks.pop()

    def _ensure_capacity(self, seq_idx: int, end_pos: int) -> None:
        """Grow block_tables[seq_idx] with fresh blocks until end_pos is covered.

        Args:
            seq_idx (int): Sequence index within the batch.
            end_pos (int): Target sequence end position.
        """
        needed_blocks = math.ceil(end_pos / self.block_size)
        table = self.block_tables[seq_idx]
        while len(table) < needed_blocks:
            table.append(self._alloc_block())

    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        """Update paged KV cache with new key and value projection tensors.

        Args:
            k (torch.Tensor): Key projection tensor of shape ``(B, H, T, D)``.
            v (torch.Tensor): Value projection tensor of shape ``(B, H, T, D)``.
            start_pos (int): Sequence starting position index.
        """
        B, H, T, D = k.shape
        end_pos = start_pos + T
        assert end_pos <= self.max_len, (
            f"KV cache capacity exceeded: start_pos={start_pos} + T={T} = {end_pos} "
            f"> cache max_len {self.max_len}"
        )
        assert k.dtype == self.pool_keys.dtype, (
            f"KV cache dtype mismatch: got {k.dtype}, cache is {self.pool_keys.dtype}."
        )

        k_store = k.detach() if not torch.is_grad_enabled() else k
        v_store = v.detach() if not torch.is_grad_enabled() else v

        for b in range(B):
            self._ensure_capacity(b, end_pos)
            table = self.block_tables[b]

            pos = start_pos
            while pos < end_pos:
                logical_block = pos // self.block_size
                offset = pos % self.block_size
                phys_block = table[logical_block]
                n = min(self.block_size - offset, end_pos - pos)
                chunk_start = pos - start_pos
                self.pool_keys[phys_block, :, offset : offset + n] = k_store[
                    b, :, chunk_start : chunk_start + n
                ]
                self.pool_values[phys_block, :, offset : offset + n] = v_store[
                    b, :, chunk_start : chunk_start + n
                ]
                pos += n

        self._B = B
        self._live = (k, v, start_pos, end_pos)

    def _gather(self, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather physical blocks into contiguous key and value tensors.

        Args:
            end_pos (int): Sequence end position index to gather up to.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Gathered Key and Value tensors ``(B, H, end_pos, D)``.
        """
        B = self._B
        n_logical = math.ceil(end_pos / self.block_size)

        idx = torch.tensor(
            [self.block_tables[b][:n_logical] for b in range(B)],
            device=self.pool_keys.device,
            dtype=torch.long,
        )  # (B, n_logical)

        flat_idx = idx.reshape(-1)  # (B * n_logical,)
        k_blocks = self.pool_keys.index_select(0, flat_idx)  # (B*n_logical, H, block_size, D)
        v_blocks = self.pool_values.index_select(0, flat_idx)

        k_blocks = k_blocks.view(B, n_logical, self.num_heads, self.block_size, self.head_dim)
        v_blocks = v_blocks.view(B, n_logical, self.num_heads, self.block_size, self.head_dim)

        # (B, n_logical, H, block, D) -> (B, H, n_logical*block, D)
        k_full = k_blocks.permute(0, 2, 1, 3, 4).reshape(
            B, self.num_heads, n_logical * self.block_size, self.head_dim
        )
        v_full = v_blocks.permute(0, 2, 1, 3, 4).reshape(
            B, self.num_heads, n_logical * self.block_size, self.head_dim
        )

        return k_full[:, :, :end_pos], v_full[:, :, :end_pos]

    def get_kv(self, start_pos: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve key and value states, preserving autograd graphs for live updates.

        Args:
            start_pos (int): Start token index.
            end_pos (int): End token index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Key and Value tensors of shape ``(B, H, S, D)``.
        """
        assert self._live is not None, "update() must be called before get_kv()"
        k_live, v_live, sp, ep = self._live
        assert sp == start_pos and ep == end_pos, (
            "get_kv() span doesn't match the most recent update() call — "
            "call update() then get_kv() with the same (start_pos, end_pos) each step."
        )

        k_full, v_full = self._gather(end_pos)
        k_full = _grad_safe_splice(k_full, k_live, start_pos, end_pos)
        v_full = _grad_safe_splice(v_full, v_live, start_pos, end_pos)
        return k_full, v_full

    def peek_kv(
        self, start_pos: int = 0, end_pos: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read-only inspection of key and value states without autograd node splicing.

        Args:
            start_pos (int, default=0): Start position.
            end_pos (int | None, default=None): Optional end position.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Inspected Key and Value tensors.
        """
        if end_pos is None:
            end_pos = self.max_len
        k_full, v_full = self._gather(end_pos)
        return k_full[:, :, start_pos:], v_full[:, :, start_pos:]

    def free_sequence(self, seq_idx: int) -> None:
        """Return physical blocks assigned to a sequence back to the free pool.

        Args:
            seq_idx (int): Target sequence index to free.
        """
        table = self.block_tables[seq_idx]
        self.free_blocks.extend(table)
        self.block_tables[seq_idx] = []

    def reset(self) -> None:
        """Reset block pool allocations and clear state tables."""
        self.free_blocks = list(range(self.num_blocks))
        self.block_tables = [[] for _ in range(self.batch_size)]
        self._live = None
        self._B = None