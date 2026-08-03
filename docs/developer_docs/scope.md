# StackFormer Scope & Design Non-Goals

This document defines StackFormer's positioning, product scope, build-vs-integrate principles, and explicit non-goals.

---

## 1. Product Positioning & Core Identity

StackFormer is positioned as a **composable PyTorch Transformer framework** designed for fast model design, experimentation, and research prototyping.

### Core Value Proposition
- **One Config, Any Block Variant**: Swapping attention variants (MHA vs GQA vs MQA), feed-forward functions (GELU vs SwiGLU vs GeGLU), normalization layers, or positional embeddings requires changing parameters in `BlockConfig`, not rewriting model files.
- **Pure PyTorch Architecture Zoo**: Standardized implementations of 10 canonical architectures (GPT-1/2, LLaMA-1/2, Gemma-1, BERT, RoBERTa, Vaswani Transformer, ViT, SegFormer-B0) that serve as runnable baselines and research starting points.
- **Zero-Bloat Training Engine**: High-level `Trainer` providing essential training primitives (AMP, DDP, gradient accumulation, clipping, SafeTensors checkpointing) without pulling in massive framework ecosystems.

---

## 2. Build vs. Integrate Engineering Principles

To maintain a clean codebase without reinventing system kernels or heavy infrastructure, StackFormer adheres to explicit build-vs-integrate boundaries (from `reviews/00_FUTURE_PLAN.md`):

| Technical Subsystem | StackFormer Strategy | Engineering Rationale |
| :--- | :--- | :--- |
| **Block Composability & Model Zoo** | 🔨 **Build Directly** | Core library identity — must be natively owned. |
| **Attention Kernels** | 🔗 **Hybrid Dispatcher** | Wraps native SDPA, FlexAttention, and `flash-attn` via `AttentionEngine`; custom Triton kernels used only for fallbacks. |
| **Distributed Sharding (FSDP2)** | 📦 **Integrate PyTorch Native** | Uses native `torch.distributed.fsdp.fully_shard` and `DTensor` rather than building custom sharding engines. |
| **Low-Precision Quantization** | 📦 **Integrate `torchao`** | Leverages `torchao` primitives for INT8/FP8/NF4 tensor quantization. |
| **HTTP Serving & MCP Server** | 🔗 **Hybrid Glue** | Uses FastAPI and standard `mcp` SDK to expose OpenAI-compatible REST and tool-calling endpoints. |
| **Production Serving Hand-off** | 📦 **Export / Convert** | Never reimplements heavy serving engines (vLLM, TensorRT-LLM); exports weights/configs to them. |

---

## 3. Explicit Non-Goals (Out of Scope)

The following items are explicitly **out of scope** for StackFormer:

1. **Reinventing C++/CUDA Sharding Engines**: StackFormer will not build custom Megatron-style or ZeRO C++ distributed engines; it standardizes on PyTorch native FSDP2 and `DTensor`.
2. **Re-implementing Monolithic Model Libraries**: StackFormer is not a drop-in replacement for Hugging Face `transformers` pretrained weight hubs (e.g. hosting thousands of fine-tuned checkpoints).
3. **Custom Low-Level CUDA Kernels**: StackFormer does not maintain custom C++/CUDA CUB attention kernels when PyTorch SDPA, FlexAttention, or FlashAttention exist.
4. **Monolithic Fine-Tuning Frameworks**: StackFormer provides lightweight native `LoRALinear`/`DoRALinear` modules on `BlockConfig` rather than wrapping heavy external fine-tuning wrappers (`peft`, `TRL`, `Axolotl`).
