# Data Ingestion Subsystem

Status: Planned, not implemented (see DESIGN.md for architecture blueprint)

## Purpose
Provides stateful streaming datasets, document token packing, and multi-worker collation primitives for pre-training, SFT, and multimodal workloads.

## Planned Contents
- Streaming iterable dataset with checkpointable resumption state.
- Zero-pad sequence packing (`PackedSequenceCollator`).
- Multimodal sample interleaving and batching.

## Checklist
- [ ] Implement `StreamingDataset` with state dict save/load for exact epoch/step resumption.
- [ ] Implement `PackedSequenceCollator` for sequence packing with cross-document attention masks.
- [ ] Integrate data state dict with `Trainer.save_checkpoint()` / `load_checkpoint()`.
- [ ] Add unit tests for dataset workers and collation determinism.
