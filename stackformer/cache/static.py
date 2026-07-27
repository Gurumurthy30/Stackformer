import torch
import torch.nn as nn
from stackformer.utils.cache import _grad_safe_splice


class StaticKVCache(nn.Module):
    """Pre-allocated (B, H, Lmax, D) buffer strategy. One instance per
    attention layer — mirrors the original kv_cache_multihead /
    kv_cache_group_query buffer behavior, just factored out behind the
    KVCachePolicy interface.
    """

    def allocate(self, batch_size, num_heads, head_dim, max_len, device, dtype, **kw):
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
        self._live = None

    def update(self, k, v, start_pos):
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

        # detach only at inference so training gradients keep flowing through
        # k_proj / v_proj for the current chunk
        k_store = k.detach() if not torch.is_grad_enabled() else k
        v_store = v.detach() if not torch.is_grad_enabled() else v

        self.cache_keys[:B, :, start_pos:end_pos] = k_store
        self.cache_values[:B, :, start_pos:end_pos] = v_store

        self._B = B
        # keep the pre-detach tensors so get_kv() can splice the live
        # autograd nodes back in over the buffer's (detached) copy
        self._live = (k, v, start_pos, end_pos)

    def get_kv(self, start_pos, end_pos):
        assert self._live is not None, "update() must be called before get_kv()"
        k_live, v_live, sp, ep = self._live
        assert sp == start_pos and ep == end_pos, (
            "get_kv() span doesn't match the most recent update() call — "
            "call update() then get_kv() with the same (start_pos, end_pos) each step."
        )

        k_full = self.cache_keys[:self._B, :, :end_pos]
        v_full = self.cache_values[:self._B, :, :end_pos]

        k_full = _grad_safe_splice(k_full, k_live, start_pos, end_pos)
        v_full = _grad_safe_splice(v_full, v_live, start_pos, end_pos)
        return k_full, v_full
    
    def peek_kv(self, start_pos: int = 0, end_pos: int = None) -> torch.Tensor:
    # read-only inspection of the buffer's first `end_pos` tokens,
    # no grad splice, no assertion tying it to the last update()
        if end_pos is None:
            end_pos = self.max_len
        return self.cache_keys[:self._B, :, start_pos:end_pos], self.cache_values[:self._B, :, start_pos:end_pos]

    def reset(self):
        self.cache_keys.zero_()
        self.cache_values.zero_()
        self._live = None