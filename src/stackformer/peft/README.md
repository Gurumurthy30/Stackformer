# Parameter-Efficient Fine-Tuning (PEFT) Subsystem

Status: Planned, not implemented

## Purpose
Provides adapter-based parameter-efficient fine-tuning techniques (LoRA, QLoRA, DoRA) implemented natively within Stackformer transformer layers without third-party dependencies.

## Planned Contents
- Low-Rank Adaptation (`LoRA`) for linear attention and feed-forward projections.
- Weight-Decomposed Low-Rank Adaptation (`DoRA`).
- Quantized LoRA (`QLoRA`) paired with 4-bit/8-bit base weights.

## Checklist
- [ ] Implement `LoRALinear` module with rank, alpha, and dropout parameters.
- [ ] Add module replacement utility injecting LoRA into `BaseModel.get_sharding_blocks()`.
- [ ] Add weight merging helper for zero-overhead inference export.
- [ ] Add Trainer fine-tuning tests verifying frozen base weights and trainable adapter gradients.
