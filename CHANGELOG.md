# Changelog

All notable changes to the StackFormer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Breaking Changes
- **Single Save Path via Trainer**: Removed `save_weights()` from `BaseModel` and model classes to eliminate duplicate/conflicting checkpointing code paths. All weight saving must now be performed via `Trainer` (`trainer.save(path)`, `trainer.save_checkpoint()`, `trainer.save_best_model()`).
- **Standalone `stackformer.load_weights`**: Checkpoint loading is centralized via `stackformer.load_weights(model, path)` and `model.load_weights(path)`. Supports direct `.safetensors`, `.pt` files, and checkpoint directories (e.g. `epoch_001_step_00100`), automatically stripping `module.` (DDP) and `_orig_mod.` (torch.compile) prefixes.

### Added
- **Documentation & README Modernization**:
  - Complete rewrite of `README.md` reflecting current library API, model zoo by category, training capabilities, and verified code snippets.
  - Added planned subsystems index `docs/PLANNED.md` and organization migration reference `docs/ORG_RENAME.md`.
- **Model Categorization and Registry**:
  - Reorganized model implementations into domain packages `stackformer.models.language` (BERT, GPT, LLaMA, Gemma, RoBERTa, Transformer, Decoder, EncoderDecoder) and `stackformer.models.vision` (ViT, SegFormer).
  - Added model registry query `stackformer.models.list_models()` and instantiation factory `stackformer.models.get_model(name, **kwargs)`.
- **Full Distributed Training Stack (Step 7)**:
  - Multi-process Distributed Data Parallel (DDP) on CPU/gloo and CUDA/NCCL.
  - Fully Sharded Data Parallel (FSDP2) via public `torch.distributed.fsdp.fully_shard`.
  - Distributed Checkpointing (DCP) support for sharded state saving and consolidation.
  - Platform guard for native Windows multi-GPU recommending WSL2.
  - Single-command multi-process launcher (`stackformer.training.launch`) setting `USE_LIBUV=0` on Windows.
- **Pretty Output Display**: `DisplayManager` supporting Rich console output with fallback to clean plain text mode.
- **Zero-Warning Code Quality Pipeline**: Ruff linting & formatting (select `E`, `F`, `W`, `I`, `B`, `UP`, `SIM`, `C4`, `RUF`), Pyright typing baseline, and Bandit security scanning.
- **Runnable Notebooks**: 5 runnable Jupyter notebooks in `examples/` covering GPT-2, BERT, ViT, SegFormer, and Seq2Seq Transformer.

### Changed
- Core dependency `safetensors>=0.4.0` moved into primary package dependencies.
- Updated `torch>=2.5,<3.0` and `numpy>=1.24,<3.0`.
- Deleted legacy TorchTitan source copies in favor of 15 clean BSD-3 licensed modules in `ops/`, `quantization/`, and `distributed/`.
- **Model Registry and Exports**: Removed non-functional scaffold classes `EncoderDecoder`, `Decoder`, and `GPTDecoder` from public language exports (#33). Updated `get_model()` to accept hyphenated architecture names such as `"gpt-2"` and `"llama-2"` (#40).
- **Generation Config Precedence**: `text_generate` now warns when explicit sampling arguments are passed alongside `generation_config` (#31).
- **Trainer Argument Validation**: `Trainer.__init__` validates keyword arguments against `TrainConfig` dataclass fields and raises `TypeError` on unknown kwargs (#36).
- **Trainer Seed Initialization**: `Trainer` applies `TrainConfig.seed` during initialization to seed Python, NumPy, PyTorch, and CUDA RNGs for training reproducibility (#34).
- **Logger Lifecycle**: `Trainer.train()` flushes loggers upon completion without closing them, allowing subsequent `Trainer.evaluate()` calls to log metrics, and `Trainer.close()` is now idempotent (#37).
- **Safetensors Load Error Handling**: `stackformer.load_weights` inspects keys using `safetensors.safe_open` and only triggers wrapper-prefix stripping fallback when prefixed keys are present, preserving strict-load error semantics (#39).

### Fixed
- **Attention Padding Masking**: Fixed BERT and RoBERTa attention mask propagation through `TransformerEncoder` and attention modules to properly mask padding tokens without evaluating tensor truthiness (#29).
- **Position Embedding Disambiguation**: Added `AbsolutePositionEmbedding.forward_ids()` for explicit position lookups, preventing token IDs from being treated as position indices (#30).
- **Pretrained Weights Stub**: `BaseModel.load_weights()` with no path now raises `NotImplementedError` rather than silently acting as a no-op (#32).
- **ViT Device & Dtype**: `ViT` now accepts and propagates `device` and `dtype` parameters to all submodules (#41).
- **RoPE Cache Bounding**: Global rotary embedding cache is now bounded to 32 entries using LRU eviction and sequence length bucketing (#42).
- **Gradient Accumulation Epoch-End Flush**: Trailing micro-batches at epoch boundaries are flushed with gradient rescaling by `accum_steps / pending`, and the accumulation counter is reset per window (#35).
- **Epoch Loss Logging**: Epoch loss is now dispatched and logged to all active loggers under the `epoch/epoch_loss` metric namespace (#37).
- **LRMonitor Double-Logging**: `LRMonitor` now updates the step metrics dictionary in place rather than creating duplicate, loss-less rows in file loggers (#37).
- **DDP Checkpoint Race Condition**: Non-DCP checkpoint saving is now gated to rank 0 with barrier synchronization across ranks, and distributed evaluation metrics are all-reduced with `dist_mean` (#38).

---

## [0.1.10] - 2026-10-31

### Added
- **Modern `src/` Layout**: Relocated core package to `src/stackformer/` to prevent packaging bugs and working directory import pollution.
- **Lowercase Module Names (PEP 8)**: Standardized `modules/attention.py`, `modules/feed_forward.py`, and `modules/normalization.py`.
- **Test Suite Expansion**: Comprehensive unit, integration, and distributed test coverage across modules, models, training, distributed launch, and vision targets.
- **Distributed & Vision Test Folders**: Organized tests into `tests/distributed/` and `tests/vision/`.
- **Complete MkDocs Documentation Website**: Comprehensive documentation ecosystem powered by Material for MkDocs, Mermaid diagrams, MathJax, and `mkdocstrings`.

### Changed
- Updated package discovery configuration in `pyproject.toml` to find packages under `src/`.
- Standardized import paths across all `src/` and `tests/` files.
- Preserved backward compatibility aliases for all PascalCase classes and legacy model names.

---

## [0.1.9] - 2026-08-25

### Added
- **Stateful KV Cache Interface (Planned)**: Specified standardized `prefill(prompt_ids)` and `decode(next_token, cache)` interface contract in `stackformer.generate` for planned model-level KV cache acceleration.
- **Fast Autoregressive Generation**: Updated `text_generate()` in `stackformer.generate` to auto-detect model KV-cache support.
- **SafeTensors Checkpointing**: Native `.safetensors` model weight serialization and loading via `CheckpointManager`.
- **Distributed Data Parallel (DDP)**: Added `wrap_model_ddp()` and `distributed_sampler()` helpers in `stackformer.distributed`.

### Changed
- Standardized `BlockConfig` parameter validation in `layer.py`.
- Standalone trainer engine decoupling and architecture focus.

---

## [0.1.0] - 2026-07-24

### Initial Release
- Initial public release of StackFormer.
- Core `BlockConfig` layer composition.
- Model zoo implementations: GPT-1, GPT-2, LLaMA-1, Gemma-1, BERT, RoBERTa, Vaswani Transformer, ViT, SegFormer.
- Initial modular Transformer core blocks.
