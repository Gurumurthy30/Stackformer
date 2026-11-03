# Stackformer Strategic Roadmap

This document outlines the architectural roadmap for Stackformer, detailing current capabilities in version 1.0, deferred modules to be reintegrated in future milestones, and feature targets for v2 and v3.

---

## 1. Version 1.0: Core Foundation & Scalable Data Parallelism (Completed)

The primary goal of Stackformer 1.0 is providing a clean, hackable transformer architecture library paired with a lean, Accelerate-style distributed `Trainer`.

### Core Capabilities
- **Model Architectures**: Clean, modular implementations of GPT-1/2, LLaMA-1/2, Gemma, BERT, RoBERTa, vanilla Transformer (seq2seq), ViT, and SegFormer sharing a unified `BlockConfig` and `BaseModel` contract.
- **Distributed Training**:
  - Native **Distributed Data Parallel (DDP)** via PyTorch `DistributedDataParallel`.
  - Native **Fully Sharded Data Parallel (FSDP2)** via PyTorch `torch.distributed.fsdp.fully_shard`.
  - Fixed transformation pipeline:
    $$\text{AC} \longrightarrow \text{compile} \longrightarrow \text{DDP/FSDP2}$$
  - Lightweight single-command distributed launcher (`stackformer.training.launch`) supporting multi-process spawning with automatic environment setup (`USE_LIBUV=0` on Windows) and rank-0 log coordination.
- **Memory & Compute Optimization**:
  - Full and selective **Activation Checkpointing (AC)** wrapping repeating blocks.
  - Integration with **`torch.compile`** at the per-block level.
  - Mixed precision training support (`bf16` on CPU/CUDA, `fp16` on CUDA).
- **Persistence & Checkpointing**:
  - Atomic single-file checkpoints (`safetensors` in core dependencies and PyTorch `.pt`).
  - PyTorch **Distributed Checkpoint (DCP)** integration across sharded ranks.
  - Mid-epoch exact resume, `latest` pointer, and automated `best/` metric tracking with pruning.
- **Observability & Presentation**:
  - Structured pretty output **`DisplayManager`** with Rich formatting, clean plain fallback, rank-0 gating, and verbosity controls (`quiet`, `normal`, `verbose`).
  - Modular metrics base (`Accuracy`, `Perplexity`, classification metrics).
  - Multi-backend loggers (Console, CSV, JSONL, TensorBoard, Weights & Biases).
  - Lifecycle callbacks (`Callback`, `EarlyStopping`, `LRMonitor`).
- **Packaging & Dependency Architecture**:
  - Single source of truth in `pyproject.toml` with `torch>=2.5,<3.0`, `numpy>=1.24,<3.0`, and `safetensors>=0.4.0` in core dependencies.
  - Modular extras: `train` (`tqdm`, `rich`), `tensorboard`, `wandb`, `float8` (`torchao`), `dev`.
  - Zero-vulnerability dependency tree verified via `pip-audit`.

---

## 2. Deferred Multi-Dimensional Parallelisms (To Be Reintegrated Later)

To keep the 1.0 codebase lean, testable, and stable across platforms (including Windows and CPU workflows), multi-dimensional model parallelisms are deferred to be reintegrated in future post-1.0 releases:

1. **Tensor Parallelism (TP)**:
   - Intra-node tensor sharding across attention and MLP projections using PyTorch's native `parallelize_module` (`ColwiseParallel`, `RowwiseParallel`).
   - Planned support across decoder layers, vision encoder projections, and MoE blocks.
2. **Pipeline Parallelism (PP)**:
   - Inter-node pipeline partitioning for linear transformer stacks (GPT, LLaMA, Gemma, BERT, ViT) using `torch.distributed.pipelining`.
   - Planned support for 1F1B and Interleaved 1F1B pipeline schedules.
3. **Context Parallelism (CP)**:
   - Ring attention and all-gather token splitting along the sequence dimension for long-context language models.

---

## 3. Post-1.0 Roadmap

### Phase 2: High-Context & Reintegrated Parallelisms
- **Reintegration of TP / PP / CP**: Native mesh coordination combining Data Parallelism with Tensor and Pipeline Parallelism.
- **Context Parallelism (CP)**: Integration of ring attention with `AttentionEngine` and FlexAttention.
- **Declarative Configuration**: Ergonomic YAML/TOML configuration layer with schema validation.

### Phase 3: Adaptation, Quantization & Frontier Scaling
- **Parameter-Efficient Fine-Tuning (PEFT)**: Native `LoRA` and `DoRA` adapters implemented directly within `BlockConfig`.
- **Low-Precision Quantization**: Native `torchao` integration (INT8, INT4 weight-only, Float8, and NF4).
- **Model Exporters**: Standardized ONNX and TorchDynamo AOT export helpers.

### Phase 4: Data Ingestion & Evaluation Benchmarks
- **Multimodal Resumable Data System**: Native streaming dataset loading with stateful resumption, token packing, and multimodal batching.
- **Evaluation Harness**: Standardized task benchmarks (perplexity, zero-shot evaluation, generation benchmarks).
- **Alignment Trainers**: Native DPO (Direct Preference Optimization) and GRPO (Group Relative Policy Optimization) trainer classes.
