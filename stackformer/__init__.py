"""Stackformer — top-level public API for the StackFormer library.

Exposes:
    - ModelConfig, GenerationConfig, TrainingConfig: Core configuration classes
    - text_generate: Text generation inference utility
    - GPT_1, GPT_2, Transformer: GPT-family and vanilla Transformer model architectures
    - Gemma_1_2B, Gemma_1_7B (gemma_1_2b, gemma_1_7b): Google Gemma model architectures
    - Llama_1, Llama_2 (llama_1, llama_2): Meta LLaMA model architectures
    - ViT, SegFormerB0: Vision Transformer and SegFormer vision models
    - BERT, RoBERTa: Encoder-based language models
    - Multi_Head_Attention, Group_query_Attention, Multi_query_Attention, Self_Attention, Cross_MultiHead_Attention: Attention modules
    - LayerNormalization, RMSNormalization: Normalization layers
    - AbsolutePositionEmbedding, SinusoidalPositionalEmbedding, RoPE: Positional embeddings
    - TransformerEncoder, TransformerDecoder, BlockConfig: Layer building blocks
    - Trainer: High-level engine trainer
"""

from .modules.masks import make_mask
from .modules.attention_engine import _run_sdpa
from .config import GenerationConfig, ModelConfig, TrainingConfig
from .engine import Trainer
from .generate import text_generate
from .language import BERT, RoBERTa
from .models import (
    GPT_1,
    GPT_2,
    Gemma_1_2B,
    Gemma_1_7B,
    Llama_1,
    Llama_2,
    Transformer,
    gemma_1_2b,
    gemma_1_7b,
    llama_1,
    llama_2,
)
from .modules import (
    AbsolutePositionEmbedding,
    BlockConfig,
    Cross_MultiHead_Attention,
    FF_GELU,
    FF_GeGLU,
    FF_LeakyReLU,
    FF_ReLU,
    FF_Sigmoid,
    FF_SiLU,
    FF_SwiGLU,
    Group_query_Attention,
    Group_query_Attention_With_RoPE,
    LayerNormalization,
    Multi_Head_Attention,
    Multi_Head_Attention_With_RoPE,
    Multi_query_Attention,
    Multi_query_Attention_With_RoPE,
    RMSNormalization,
    RoPE,
    Self_Attention,
    SinusoidalPositionalEmbedding,
    TransformerDecoder,
    TransformerEncoder,
    kv_cache_group_query,
    kv_cache_multihead,
)
from .vision import SegFormerB0, ViT

__all__ = [
    "AbsolutePositionEmbedding",
    "BlockConfig",
    "BERT",
    "Cross_MultiHead_Attention",
    "FF_GELU",
    "FF_GeGLU",
    "FF_LeakyReLU",
    "FF_ReLU",
    "FF_Sigmoid",
    "FF_SiLU",
    "FF_SwiGLU",
    "GPT_1",
    "GPT_2",
    "Gemma_1_2B",
    "Gemma_1_7B",
    "Group_query_Attention",
    "Group_query_Attention_With_RoPE",
    "LayerNormalization",
    "Llama_1",
    "Llama_2",
    "Multi_Head_Attention",
    "Multi_Head_Attention_With_RoPE",
    "Multi_query_Attention",
    "Multi_query_Attention_With_RoPE",
    "_run_sdpa",
    "RMSNormalization",
    "RoPE",
    "RoBERTa",
    "SegFormerB0",
    "Self_Attention",
    "SinusoidalPositionalEmbedding",
    "TransformerEncoder",
    "TransformerDecoder",
    "ViT",
    "gemma_1_2b",
    "gemma_1_7b",
    "kv_cache_group_query",
    "kv_cache_multihead",
    "llama_1",
    "llama_2",
    "make_mask",
    "text_generate",
    "Transformer",
    "GenerationConfig",
    "ModelConfig",
    "TrainingConfig",
    "Trainer",
    "make_mask"
]

