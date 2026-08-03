# StackFormer Technical Roadmap

> Source: `reviews/00_FUTURE_PLAN.md` · Deep Technical Roadmap Specification

Development on StackFormer follows a linear, sequential engineering execution order across 10 technical tracks. The cross-track sequence prioritizes hardening training, architecture, and research capabilities before building high-concurrency inference and serving infrastructure.

---

## Technical Tracks Overview

1. **KV Cache Subsystem**: Static, Paged, Quantized (FP8), and Streaming KV caches.
2. **Distributed Training**: Native PyTorch FSDP2 sharding (`fully_shard`) + `DTensor` + Distributed Checkpoint (`DCP`).
3. **Fine-Tuning & Alignment**: Native `LoRALinear` / `DoRALinear` adapters, DPO, GRPO / RLAIF alignment trainers.
4. **Quantization & Compression**: `torchao` low-precision primitives (INT8, INT4, FP8, NF4).
5. **Multimodal & Document AI**: Hierarchical vision encoders (SegFormer/ViT) + MLP projectors + LLaMA decoder stacks.
6. **Attention Engine & Modular Composition**: Unified `AttentionEngine` kernel dispatcher (SDPA → FlexAttention → FlashAttn v2/v4 → Triton) with zero `torch.compile` graph breaks.
7. **Execution & Budget Planner**: Analytical VRAM/execution planner (`plan_training()`).
8. **Inference & Serving Engine**: PagedAttention virtual block tables, continuous batching queue, FastAPI REST server.
9. **Tool Integration & MCP**: Native Model Context Protocol (MCP) tool-calling server.
10. **Model Export**: Conversion to vLLM, TensorRT-LLM, ONNX Runtime, and GGUF formats.

---

## Execution Phases & Deliverables

### Phase 0 — Unblock Core Generation & Fix KV-Cache Contract (Completed / Current)
- **Goal**: Resolve the KV-cache contract gap in autoregressive generation.
- **Deliverables**:
  - Implement standardized `prefill(prompt_ids)` and `decode(next_token, cache)` methods on `Llama2` (`stackformer/models/llama.py`).
  - Update `text_generate()` in `stackformer/generate.py` to route through model `prefill()`/`decode()` when implemented.
  - Rewrite `tests/models/test_kv_cache_generation_parity.py` to validate incremental decoding parity against full context re-computation.

---

### Phase 1 — Attention Engine & Compiler Integration (Planned)
- **Goal**: Implement `AttentionEngine` unified kernel dispatcher in `stackformer/modules/attention_engine.py`.
- **Deliverables**:
  - Dynamic routing across backends: SDPA (`F.scaled_dot_product_attention`), FlexAttention (`torch.nn.attention.flex_attention`), `flash-attn` v2/v4, and custom Triton fallback.
  - Functional mask mods (`sliding_window`, `dilated`, `mistral`, `document_mask`) for FlexAttention score modification.
  - Zero graph-break validation under `torch.compile`.

---

### Phase 2 — Budget Planner & PyTorch Native FSDP2 Scaling (Planned)
- **Goal**: Analytical training planner and cluster-scale distributed training on native PyTorch primitives.
- **Deliverables**:
  - `plan_training(model_config, GPU_spec)` in `stackformer/planner/budget.py` producing VRAM, batch size, and sharding recommendations.
  - Standardize distributed training on PyTorch FSDP2 (`torch.distributed.fsdp.fully_shard`) + `DTensor`.
  - PyTorch Distributed Checkpoint (`DCP`) integration in `CheckpointManager`.

---

### Phase 3 — `torchao` Quantization & Native Adapters (Planned)
- **Goal**: Low-precision training/inference and zero-bloat parameter-efficient fine-tuning.
- **Deliverables**:
  - Standardize quantization on `torchao` (`int8_weight_only`, `int4_weight_only`, `float8`, `nf4`).
  - Build native `LoRALinear` and `DoRALinear` modules into `BlockConfig` (`stackformer/peft/`).
  - Implement `DPOTrainer` and `GRPOTrainer` for alignment post-training.

---

### Phase 4 — Multimodal Vision-Language & Document AI (Planned)
- **Goal**: Combine vision encoders with language decoder backbones.
- **Deliverables**:
  - Reuse SegFormer-B0 and ViT patch backbones from `stackformer/vision/`.
  - Build linear, MLP, and Spatial Resampler projectors in `stackformer/multimodal/projectors.py`.
  - Assemble multimodal document AI reference model (`multimodal_doc.py`).

---

### Phase 5 — Paged KV Cache & Serving Engine (Planned)
- **Goal**: High-concurrency production serving infrastructure once training/research foundation is complete.
- **Deliverables**:
  - `PagedKVCache` with virtual block tables (block size 16) in `stackformer/cache/paged.py`.
  - `KVCacheManager` orchestrating dynamic page allocation.
  - Continuous batching request scheduler and OpenAI-compatible FastAPI server (`/v1/chat/completions`).
  - Model Context Protocol (MCP) stdio tool-calling server (`stackformer/mcp/server.py`).
