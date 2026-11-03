"""Modules package containing reusable Transformer building blocks for StackFormer.

Exposes:
    - Multi_Head_Attention, Group_query_Attention, Multi_query_Attention, Self_Attention, Cross_MultiHead_Attention: Attention components
    - kv_cache_multihead, kv_cache_group_query: Stateful KV-cache attention implementations
    - FF_GELU, FF_ReLU, FF_SwiGLU, FF_GeGLU, FF_SiLU, FF_LeakyReLU, FF_Sigmoid: Feed-forward networks
    - LayerNormalization, RMSNormalization: Normalization layers
    - AbsolutePositionEmbedding, SinusoidalPositionalEmbedding, RoPE: Position embeddings
    - make_mask: Causal, padding, and sliding window mask generator
    - BlockConfig, TransformerEncoder, TransformerDecoder: High-level block and backbone layers
"""

from .attention import (
    Cross_MultiHead_Attention,
    CrossMultiHeadAttention,
    Group_query_Attention,
    Group_query_Attention_With_RoPE,
    GroupQueryAttention,
    GroupQueryAttentionWithRoPE,
    KVCacheGroupQuery,
    KVCacheMultiHead,
    Multi_Head_Attention,
    Multi_Head_Attention_With_RoPE,
    Multi_query_Attention,
    Multi_query_Attention_With_RoPE,
    MultiHeadAttention,
    MultiHeadAttentionWithRoPE,
    MultiQueryAttention,
    MultiQueryAttentionWithRoPE,
    Self_Attention,
    # New canonical names
    SelfAttention,
    kv_cache_group_query,
    kv_cache_multihead,
)
from .feed_forward import (
    FF_GELU,
    FeedForwardGeGLU,
    FeedForwardGELU,
    FeedForwardLeakyReLU,
    FeedForwardReLU,
    FeedForwardSigmoid,
    FeedForwardSiLU,
    FeedForwardSwiGLU,
    FF_GeGLU,
    FF_LeakyReLU,
    FF_ReLU,
    FF_Sigmoid,
    FF_SiLU,
    FF_SwiGLU,
)
from .layer import (
    BlockConfig,
    TransformerDecoder,
    TransformerEncoder,
)
from .normalization import LayerNorm, RMSNorm
from .position_embedding import (
    AbsolutePositionEmbedding,
    RoPE,
    SinusoidalPositionalEmbedding,
)

# Backward-compat aliases
LayerNormalization = LayerNorm
RMSNormalization = RMSNorm


__all__ = [
    "AbsolutePositionEmbedding",
    "BlockConfig",
    "Cross_MultiHead_Attention",
    "FF_GELU",
    "FF_GeGLU",
    "FF_LeakyReLU",
    "FF_ReLU",
    "FF_Sigmoid",
    "FF_SiLU",
    "FF_SwiGLU",
    "FeedForwardReLU",
    "FeedForwardLeakyReLU",
    "FeedForwardGELU",
    "FeedForwardSigmoid",
    "FeedForwardSiLU",
    "FeedForwardSwiGLU",
    "FeedForwardGeGLU",
    "Group_query_Attention",
    "Group_query_Attention_With_RoPE",
    "LayerNorm",
    "LayerNormalization",
    "Multi_Head_Attention",
    "Multi_Head_Attention_With_RoPE",
    "Multi_query_Attention",
    "Multi_query_Attention_With_RoPE",
    "RMSNorm",
    "RMSNormalization",
    "RoPE",
    "Self_Attention",
    "SinusoidalPositionalEmbedding",
    "TransformerEncoder",
    "TransformerDecoder",
    "kv_cache_group_query",
    "kv_cache_multihead",
]
