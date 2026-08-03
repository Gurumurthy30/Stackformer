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

from .Attention import (
    Cross_MultiHead_Attention,
    Group_query_Attention,
    Group_query_Attention_With_RoPE,
    Multi_Head_Attention,
    Multi_Head_Attention_With_RoPE,
    Multi_query_Attention,
    Multi_query_Attention_With_RoPE,
    Self_Attention,
    kv_cache_group_query,
    kv_cache_multihead,
)
from .Feed_forward import (
    FF_GELU,
    FF_GeGLU,
    FF_LeakyReLU,
    FF_ReLU,
    FF_Sigmoid,
    FF_SiLU,
    FF_SwiGLU,
)

from .Normalization import LayerNormalization, RMSNormalization
from .layer import (
    BlockConfig,
    TransformerDecoder,
    TransformerEncoder,
)
from .position_embedding import (
    AbsolutePositionEmbedding,
    RoPE,
    SinusoidalPositionalEmbedding,
)


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
    "Group_query_Attention",
    "Group_query_Attention_With_RoPE",
    "LayerNormalization",
    "Multi_Head_Attention",
    "Multi_Head_Attention_With_RoPE",
    "Multi_query_Attention",
    "Multi_query_Attention_With_RoPE",
    "RMSNormalization",
    "RoPE",
    "Self_Attention",
    "SinusoidalPositionalEmbedding",
    "TransformerEncoder",
    "TransformerDecoder",
    "kv_cache_group_query",
    "kv_cache_multihead",
]