# StackFormer Developer Architecture Specification

This document details the internal software architecture of StackFormer, covering module hierarchy, `BlockConfig` resolution, pure PyTorch interoperability, and KV-cache generation contracts.

---

## 1. Modular Architecture Principles

StackFormer is designed around three core architectural tenets:

1. **Orthogonal Layer Primitives**: Attention, feed-forward, normalization, and positional embedding components are decoupled modules residing under `stackformer.modules`.
2. **Declarative Block Composition**: `BlockConfig` encapsulates all layer choices into a single configuration dataclass. High-level blocks (`EncoderBlock`, `DecoderBlock`, `TransformerEncoder`, `TransformerDecoder`) inspect `BlockConfig` to instantiate matching submodules using private factory functions (`_build_attention`, `_build_ffn`, `_build_norm`).
3. **Pure PyTorch Execution**: Avoids custom C++ compile steps in core building blocks, relying on PyTorch's native `nn.Module`, Scaled Dot-Product Attention (`F.scaled_dot_product_attention`), and autograd engine.

---

## 2. Block Configuration & Factory Resolution

### `BlockConfig` Pipeline

When `TransformerEncoder(config, num_layers=N)` is instantiated:

```text
               ┌───────────────────────┐
               │      BlockConfig      │
               └──────────┬────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   _build_attention   _build_norm    _build_ffn
          │               │               │
  [MHA / GQA / MQA]  [LayerNorm /   [GELU / SwiGLU /
   + [RoPE / Abs]      RMSNorm]      GeGLU / ReLU]
          │               │               │
          └───────────────┼───────────────┘
                          ▼
               ┌───────────────────────┐
               │     EncoderBlock      │
               └───────────────────────┘
```

- `_build_attention`: Resolves `"mha"`, `"gqa"`, `"mqa"`, `"cross_mha"`, and RoPE variants (`"mha_rope"`, `"gqa_rope"`, `"mqa_rope"`).
- `_build_ffn`: Resolves `"gelu"`, `"swiglu"`, `"geglu"`, `"leaky_relu"`, `"relu"`, `"sigmoid"`, `"silu"`.
- `_build_norm`: Resolves `"layernorm"` (`nn.LayerNorm`) and `"rmsnorm"` (`stackformer.modules.RMSNorm`).

---

## 3. Training Interoperability

StackFormer focuses on composable architecture blocks, positional encodings, attention mechanisms, and generation caches. All models are standard `torch.nn.Module` objects:

1. **Standard PyTorch Loops**: Can be optimized using standard PyTorch optimizers (`torch.optim.AdamW`, `torch.optim.SGD`) and standard training loops.
2. **Third-Party Trainer Ecosystem**: Designed to interoperate cleanly with external training frameworks such as Hugging Face Trainer, PyTorch Lightning, Accelerate, and torchtune.
3. **Standalone Trainer Library**: Specialized high-level training orchestrator logic is maintained separately, keeping the core StackFormer modeling library clean and lightweight.

---

## 4. KV-Cache & Text Generation Contract

StackFormer autoregressive generation in `stackformer.generate.text_generate()` implements a dual-path mechanism:

1. **Explicit Method Contract (`prefill` and `decode`)**:
   - Model implements `prefill(prompt_ids) -> (logits, cache_dict)`
   - Model implements `decode(next_token, cache_dict) -> (logits, updated_cache_dict)`
   - `Llama2` implements this contract using stateful `kv_cache_group_query` modules.

2. **Context Re-computation Fallback**:
   - For models without `prefill()`/`decode()` (or where `supports_kv_cache = False`), `text_generate()` re-runs the full context window slice `generated[:, -max_context_len:]` through model `.forward()` at every token step.
