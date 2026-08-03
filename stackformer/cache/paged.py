import math
import torch
import torch.nn as nn
from stackformer.utils.cache import _grad_safe_splice


class PagedKVCache(nn.Module):
    """Block-pool KV cache. One instance per attention layer, same as
    StaticKVCache — but the physical storage is a shared pool of fixed-size
    blocks, and each sequence in the batch holds a `block_table` (list of
    physical block indices) instead of owning a private (Lmax, D) span.

    Gather-based get_kv(): materializes a contiguous (B, H, S, D) view from
    scattered blocks via index_select before calling SDPA. Correct, not a
    fused kernel — see module docstring in paged.py notes for the
    real-vLLM alternative.
    """

    def allocate( self, batch_size, num_heads, head_dim, max_len, device, dtype, 
                 block_size=16, num_blocks=None, **kw,):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.batch_size = batch_size

        blocks_per_seq_worst_case = math.ceil(max_len / block_size)
        if num_blocks is None:
            # worst case == static allocation (every seq uses its own fullspan); 
            # pass num_blocks explicitly to actually exploit sharing
            # across sequences of different lengths.
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
        # block_tables[b] = list[int] of physical block indices, logical order
        self.block_tables = [[] for _ in range(batch_size)]
        self._live = None
        self._B = None

    def _alloc_block(self):
        assert self.free_blocks, "PagedKVCache: out of physical blocks (pool exhausted)"
        return self.free_blocks.pop()

    def _ensure_capacity(self, seq_idx, end_pos):
        """Grow block_tables[seq_idx] with fresh blocks until it can hold
        tokens up to end_pos (exclusive)."""
        needed_blocks = math.ceil(end_pos / self.block_size)
        table = self.block_tables[seq_idx]
        while len(table) < needed_blocks:
            table.append(self._alloc_block())

    def update(self, k, v, start_pos):
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

            # write token-by-token-range into whichever physical blocks the
            # [start_pos, end_pos) span touches; a chunk can straddle a
            # block boundary so we walk block-by-block.
            pos = start_pos
            while pos < end_pos:
                logical_block = pos // self.block_size
                offset = pos % self.block_size
                phys_block = table[logical_block]
                n = min(self.block_size - offset, end_pos - pos)
                chunk_start = pos - start_pos
                self.pool_keys[phys_block, :, offset:offset + n] = (
                    k_store[b, :, chunk_start:chunk_start + n].transpose(0, 1)
                    if False else k_store[b, :, chunk_start:chunk_start + n]
                )
                self.pool_values[phys_block, :, offset:offset + n] = (
                    v_store[b, :, chunk_start:chunk_start + n]
                )
                pos += n

        self._B = B
        self._live = (k, v, start_pos, end_pos)

    def _gather(self, end_pos):
        """Build a contiguous (B, H, end_pos, D) tensor from scattered
        physical blocks via index_select, then trim to end_pos (last block
        may be only partially used)."""
        B = self._B
        n_logical = math.ceil(end_pos / self.block_size)

        # (B, n_logical) physical block ids
        idx = torch.tensor(
            [self.block_tables[b][:n_logical] for b in range(B)],
            device=self.pool_keys.device,
            dtype=torch.long,
        )  # (B, n_logical)

        flat_idx = idx.reshape(-1)  # (B * n_logical,)
        k_blocks = self.pool_keys.index_select(0, flat_idx)   # (B*n_logical, H, block_size, D)
        v_blocks = self.pool_values.index_select(0, flat_idx)

        k_blocks = k_blocks.view(B, n_logical, self.num_heads, self.block_size, self.head_dim)
        v_blocks = v_blocks.view(B, n_logical, self.num_heads, self.block_size, self.head_dim)

        # (B, n_logical, H, block, D) -> (B, H, n_logical*block, D)
        k_full = k_blocks.permute(0, 2, 1, 3, 4).reshape(B, self.num_heads, n_logical * self.block_size, self.head_dim)
        v_full = v_blocks.permute(0, 2, 1, 3, 4).reshape(B, self.num_heads, n_logical * self.block_size, self.head_dim)

        return k_full[:, :, :end_pos], v_full[:, :, :end_pos]

    def get_kv(self, start_pos, end_pos):
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

    def peek_kv(self, start_pos: int = 0, end_pos: int = None):
        if end_pos is None:
            end_pos = self.max_len
        k_full, v_full = self._gather(end_pos)
        return k_full[:, :, start_pos:], v_full[:, :, start_pos:]

    def free_sequence(self, seq_idx):
        """Return one sequence's blocks to the free list without touching
        the others — the thing static allocation can't do cheaply."""
        table = self.block_tables[seq_idx]
        self.free_blocks.extend(table)
        self.block_tables[seq_idx] = []

    def reset(self):
        self.free_blocks = list(range(self.num_blocks))
        self.block_tables = [[] for _ in range(self.batch_size)]
        self._live = None
        self._B = None