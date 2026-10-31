# Changelog

All notable changes to the StackFormer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.10] - 2026-10-31

### Added
- **Modern `src/` Layout**: Relocated core package to `src/stackformer/` to prevent packaging bugs and working directory import pollution.
- **Lowercase Module Names (PEP 8)**: Standardized `modules/attention.py`, `modules/feed_forward.py`, and `modules/normalization.py`.
- **Test Suite Expansion**: Comprehensive unit and integration test coverage expanded to **153 verified tests** across modules, models, caching, trainer, distributed, and vision targets.
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
- Refactored `Trainer` engine loop with gradient accumulation and AMP scaling.

---

## [0.1.0] - 2026-07-24

### Initial Release
- Initial public release of StackFormer.
- Core `BlockConfig` layer composition.
- Model zoo implementations: GPT-1, GPT-2, LLaMA-1, Gemma-1, BERT, RoBERTa, Vaswani Transformer, ViT, SegFormer.
- Basic `Trainer` and optimizer factories.

