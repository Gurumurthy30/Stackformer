"""Key-value cache strategies and buffer management for StackFormer.

Exposes:
    - StaticKVCache: Pre-allocated contiguous buffer KV cache
    - PagedKVCache: Memory-efficient block-pool KV cache
"""

from .paged import PagedKVCache
from .static import StaticKVCache

__all__ = ["StaticKVCache", "PagedKVCache"]