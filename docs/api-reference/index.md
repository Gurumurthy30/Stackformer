# StackFormer API Reference

This document details the public API surface exposed by `stackformer` and its core modules.

---

## Configuration Dataclasses (`stackformer.config`)

### `ModelConfig`
Container for model hyperparameter definitions.
- **Attributes**: `vocab_size` (int), `embed_dim` (int), `num_layers` (int), `num_heads` (int), `seq_len` (int), `hidden_dim` (int), `dropout` (float = 0.0).

### `TrainingConfig`
Container for optimization and training loop parameters.
- **Attributes**: `max_epochs` (int = 1), `max_train_steps` (int | None), `max_eval_steps` (int | None), `eval_every_n_epochs` (int = 1), `save_every_n_epochs` (int = 1), `grad_accumulation_step` (int = 1), `max_grad_norm` (float | None), `lr` (float = 3e-4), `weight_decay` (float = 0.01), `optimizer_name` (str = "adamw"), `scheduler_name` (str = "none"), `warmup_steps` (int = 0).

### `GenerationConfig`
Container for text decoding parameters.
- **Attributes**: `max_context_len` (int = 128), `max_new_tokens` (int = 50), `temperature` (float = 1.0), `top_k` (int | None = None), `top_p` (float = 1.0), `eos_token_id` (int | None = None).

---

## Block Building Modules (`stackformer.modules`)

### `BlockConfig`
Declarative spec object passed to `EncoderBlock`, `DecoderBlock`, `TransformerEncoder`, `TransformerDecoder`.
- **Key Parameters**: `embed_dim`, `num_heads`, `num_kv_heads`, `hidden_dim`, `attention` (str), `ffn` (str), `norm` (str), `pre_norm` (bool), `dropout` (float), `qkv_bias` (bool), `device`, `dtype`.

### Attention Classes
- `SelfAttention`: Standard single-head self-attention.
- `MultiHeadAttention`: Multi-Head Attention (MHA).
- `GroupQueryAttention`: Grouped-Query Attention (GQA).
- `MultiQueryAttention`: Multi-Query Attention (MQA).
- `CrossMultiHeadAttention`: Cross-attention block for encoder-decoder models.
- RoPE variants: `MultiHeadAttentionWithRoPE`, `GroupQueryAttentionWithRoPE`, `MultiQueryAttentionWithRoPE`.
- KV-Cache stateful functions: `kv_cache_multihead`, `kv_cache_group_query`.

### Feed-Forward Networks (FFN)
- `FF_ReLU` / `FeedForwardReLU`
- `FF_LeakyReLU` / `FeedForwardLeakyReLU`
- `FF_GELU` / `FeedForwardGELU`
- `FF_Sigmoid` / `FeedForwardSigmoid`
- `FF_SiLU` / `FeedForwardSiLU`
- `FF_SwiGLU` / `FeedForwardSwiGLU`
- `FF_GeGLU` / `FeedForwardGeGLU`

### Normalization Layers
- `LayerNorm` / `LayerNormalization`
- `RMSNorm` / `RMSNormalization`

### Positional Embeddings
- `AbsolutePositionEmbedding`: Learned absolute positional embeddings lookup.
- `SinusoidalPositionalEmbedding`: Fixed sinusoidal positional encoding (Vaswani style).
- `RoPE`: Rotary Position Embedding helper and frequency cache provider.

---

## Model Zoo (`stackformer.models` & `stackformer.vision`)

- `GPT1` / `GPT2` (Aliases: `GPT_1`, `GPT_2`)
- `Llama1` / `Llama2` (Aliases: `llama_1`, `llama_2`, `Llama_1`, `Llama_2`)
- `Gemma1_2B` / `Gemma1_7B` (Aliases: `gemma_1_2b`, `gemma_1_7b`)
- `BERT` / `RoBERTa`
- `Transformer` (Canonical Vaswani Encoder-Decoder)
- `ViT` (Vision Transformer for image classification)
- `SegFormerB0` (Hierarchical SegFormer for semantic segmentation)

---

## Engine & Training (`stackformer.engine`)

### `Trainer`
High-level orchestration manager.
- **Constructor Arguments**: `model`, `train_dataloader`, `val_dataloader=None`, `optimizer=None`, `criterion=None`, `scheduler=None`, `device="cpu"`, `use_amp=False`, `use_ddp=False`, `max_epochs=1`, `max_train_steps=None`, `max_eval_steps=None`, `grad_accumulation_step=1`, `max_grad_norm=None`, `checkpoint_dir="checkpoints"`, `logger=None`.
- **Methods**: `fit() -> dict`, `save_checkpoint(path)`, `load_checkpoint(path)`.

---

## Text Generation Engine (`stackformer.generate`)

### `text_generate(model, prompt_ids, ...)`
Autoregressively decodes next tokens from prompt input.
- **Parameters**: `model` (nn.Module), `prompt_ids` (Tensor), `max_context_len` (int), `max_new_tokens` (int), `temperature` (float), `top_k` (int | None), `top_p` (float), `eos_token_id` (int | None).
- **Behavior**: Uses model `.prefill()` and `.decode()` KV-cache methods if implemented; falls back to context re-computation forward pass otherwise.
