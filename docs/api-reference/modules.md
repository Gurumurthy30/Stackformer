# Core Modules API Reference

This page documents the modular Transformer building blocks provided in `stackformer.modules`.

---

## Transformer Layers & Stacks

::: stackformer.modules.layer
    options:
      show_root_heading: true
      members:
        - BlockConfig
        - EncoderBlock
        - DecoderBlock
        - TransformerEncoder
        - TransformerDecoder

---

## Attention Mechanisms

::: stackformer.modules.attention
    options:
      show_root_heading: true
      members:
        - SelfAttention
        - MultiHeadAttention
        - GroupQueryAttention
        - MultiQueryAttention
        - CrossMultiHeadAttention
        - MultiHeadAttentionWithRoPE
        - GroupQueryAttentionWithRoPE
        - MultiQueryAttentionWithRoPE

---

## Feed-Forward Networks (FFN)

::: stackformer.modules.feed_forward
    options:
      show_root_heading: true
      members:
        - FeedForwardReLU
        - FeedForwardLeakyReLU
        - FeedForwardGELU
        - FeedForwardSigmoid
        - FeedForwardSiLU
        - FeedForwardSwiGLU
        - FeedForwardGeGLU

---

## Normalization Layers

::: stackformer.modules.normalization
    options:
      show_root_heading: true
      members:
        - LayerNorm
        - RMSNorm

---

## Positional Embeddings

::: stackformer.modules.position_embedding
    options:
      show_root_heading: true
      members:
        - AbsolutePositionEmbedding
        - SinusoidalPositionalEmbedding
        - RoPE

