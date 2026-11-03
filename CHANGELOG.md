# Changelog

All notable changes to the StackFormer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Breaking Changes
- **Single Save Path via Trainer**: Removed `save_weights()` from `BaseModel` and model classes to eliminate duplicate/conflicting checkpointing code paths. All weight saving must now be performed via `Trainer` (`trainer.save(path)`, `trainer.save_checkpoint()`, `trainer.save_best_model()`).
- **Standalone `stackformer.load_weights`**: Checkpoint loading is centralized via `stackformer.load_weights(model, path)` and `model.load_weights(path)`. Supports direct `.safetensors`, `.pt` files, and checkpoint directories (e.g. `step_00000100`), automatically stripping `module.` (DDP) and `_orig_mod.` (torch.compile) prefixes.

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

---

## [0.1.10] - 2026-10-31

### Added
- **Modern `src/` Layout**: Relocated core package to `src/stackformer/` to prevent packaging bugs and working directory import pollution.
- **Lowercase Module Names (PEP 8)**: Standardized `modules/attention.py`, `modules/feed_forward.py`, and `modules/normalization.py`.
- **Test Suite Expansion**: Comprehensive unit and integration test coverage expanded to **153 verified tests** across modules, models, caching, and vision targets.
- **Distributed & Vision Test Folders**: Organized tests into `tests/distributed/` and `tests/vision/`.
- **Complete MkDocs Documentation Website**: Comprehensive documentation ecosystem powered by Material for MkDocs, Mermaid diagrams, MathJax, and `mkdocstrings`.

### Changed
- Updated package discovery configuration in `pyproject.toml` to find packages under `src/`.
- Standardized import paths across all `src/` and `tests/` files.
- Preserved backward compatibility aliases for all PascalCase classes and legacy model names.

---

## [0.1.9] - 2026-08-25

### Added
- **LLaMA-2 Stateful KV Cache**: Implemented standardized `prefill(prompt_ids)` and `decode(next_token, cache)` methods on `Llama2` with Grouped-Query Attention (GQA).
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
