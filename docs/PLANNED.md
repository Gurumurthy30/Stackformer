# Stackformer Planned Work and Placeholder Directory Index

This document catalogs all placeholder directories, planned architectural subsystems, and forward-looking modules in Stackformer, mapping each to its status and target milestone in [ROADMAP.md](ROADMAP.md).

---

## Placeholder Index

| Directory Path | Purpose | Current Status | Target Phase |
| :--- | :--- | :--- | :--- |
| `src/stackformer/models/multimodal/` | Vision-language (e.g. CLIP, LLaVA, PaLI) and document AI architectures combining visual encoders with autoregressive decoders | Planned, not implemented | Phase 4 |
| `src/stackformer/models/audio/` | Speech recognition (ASR, Whisper), synthesis (TTS), and audio representation transformers | Planned, not implemented | Phase 4 |
| `src/stackformer/data/` | Stateful streaming datasets, token packing (`PackedSequenceCollator`), and multi-worker collation | Planned, not implemented (see [DESIGN.md](../src/stackformer/data/DESIGN.md)) | Phase 4 |
| `src/stackformer/peft/` | Parameter-Efficient Fine-Tuning adapters (`LoRA`, `QLoRA`, `DoRA`) natively integrated into transformer layers | Planned, not implemented | Phase 3 |
| `src/stackformer/export/` | Model export and deployment serialization targeting ONNX Runtime and `torch.export` (AOTInductor) | Planned, not implemented | Phase 3 |
| `src/stackformer/eval/` | Standardized task benchmarks (zero-shot, log-likelihood, generation metrics, and mIoU evaluation) | Planned, not implemented | Phase 4 |
| `src/stackformer/serving/` | Continuous-batching inference server integrating Stackformer's paged KV cache (`stackformer.cache.paged`) | Planned, not implemented | Phase 4 |
| `src/stackformer/quantization/` | Low-precision representations and transforms (FP8 GEMMs available via `torchao`, INT8/INT4 planned) | Partially implemented (Float8 active; INT8/INT4 planned) | Phase 3 |
| `benchmarks/` | Reproducible throughput (tokens/s), peak memory, and distributed scaling benchmark scripts | Planned, not implemented | Phase 4 |
| `docs/tutorials/` | End-to-end practical tutorials (pre-training, semantic segmentation, distributed training, deployment) | Planned, not implemented | Phase 4 |
| `scripts/` | Repository automation and verification utilities (e.g., link integrity checker) | Active development utilities | Phase 1 / Active |

---

## Architecture Guidelines for Placeholders

1. **No Fake Public APIs**: Placeholders contain clear `README.md` documentation and empty or minimal docstring-only `__init__.py` files. They do not expose stub functions or classes that raise `NotImplementedError` at runtime.
2. **Clear Roadmap Linkage**: Each planned subsystem corresponds directly to goals defined in [ROADMAP.md](ROADMAP.md).
3. **Additive Evolution**: Future implementations in these directories must adhere to Stackformer's zero-bloat principles, strict code quality pipelines, and platform portability.
