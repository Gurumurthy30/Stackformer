# Parameter-Efficient Fine-Tuning (PEFT)

## Overview
The `stackformer.peft` module provides interfaces for adapting pre-trained Stackformer architectures with minimal parameter updates, lower VRAM requirements, and composable adapter weights.

## Planned Capabilities
- **LoRA (Low-Rank Adaptation)**: Injects rank $r$ trainable decomposition matrices $B \times A$ alongside frozen linear projections ($W + \Delta W$).
- **Weight Tying & Sharding Preservation**: Integrates cleanly with `BaseModel.get_sharding_blocks()` and `get_tied_params()` so adapters can be trained under DDP and FSDP2 without sharding conflicts.
- **Adapter Saving & Merging**: Export standalone adapter weights (`adapter_model.safetensors`) or fold adapters back into base weights for zero-overhead inference.

