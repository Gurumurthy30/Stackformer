# StackFormer Developer Architecture Specification

This document details the internal software architecture of StackFormer, covering module hierarchy, `BlockConfig` resolution, engine/trainer mechanics, checkpoint serialization, and KV-cache generation contracts.

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

## 3. Training & Engine Subsystem (`stackformer.engine`)

The training engine splits responsibilities between state management, step execution, and high-level training loops:

```text
┌─────────────────────────────────────────────────────────────┐
│                          Trainer                            │
│  (Manages epochs, DDP setup, AMP scaling, LR scheduling)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                          Engine                             │
│  (Executes _train_step, _eval_step, computes loss & metrics)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      TrainingState                          │
│  (Tracks global steps, current epoch, loss history, device) │
└─────────────────────────────────────────────────────────────┘
```

- **AMP Integration (`stackformer.amp.scaler`)**: `AMPScaler` wraps `torch.cuda.amp.GradScaler` (or `torch.amp.GradScaler` on PyTorch 2.1+), auto-disabling silently when CUDA is unavailable.
- **Distributed Data Parallel (`stackformer.distributed.ddp`)**: `wrap_model_ddp()` wraps models with `DistributedDataParallel`, creating standard PyTorch process groups via `init_distributed()`.
- **SafeTensors Checkpointing (`stackformer.engine.checkpoint`)**: `CheckpointManager` serializes model weights to `.safetensors` format alongside a JSON file containing optimizer state, scheduler state, and training configuration.

---

## 4. KV-Cache & Text Generation Contract

StackFormer autoregressive generation in `stackformer.generate.text_generate()` implements a dual-path mechanism:

1. **Explicit Method Contract (`prefill` and `decode`)**:
   - Model implements `prefill(prompt_ids) -> (logits, cache_dict)`
   - Model implements `decode(next_token, cache_dict) -> (logits, updated_cache_dict)`
   - `Llama2` implements this contract using stateful `kv_cache_group_query` modules.

2. **Context Re-computation Fallback**:
   - For models without `prefill()`/`decode()` (or where `supports_kv_cache = False`), `text_generate()` re-runs the full context window slice `generated[:, -max_context_len:]` through model `.forward()` at every token step.
