# Quantization Subsystem

Status: Partially implemented (Float8 available; INT8/INT4 planned)

## Purpose
Provides low-precision module representations and quantization transforms to accelerate training and reduce inference memory footprints.

## Current Capabilities
- **Float8 Training**: `Float8Linear` module supporting FP8 GEMMs via `torchao` optional extra (`pip install "stackformer[float8]"`).
- **FSDP Tensor Hooks**: Distributed tensor sharding hooks for quantized parameters.

## Planned Contents
- INT8 weight-only and dynamic activation quantization.
- INT4 / NF4 quantization for parameter-efficient fine-tuning (QLoRA).
- Quantization-Aware Training (QAT) modules.

## Checklist
- [x] Implement `Float8Linear` module with FSDP2 sharding support.
- [x] Configure `[float8]` extra in `pyproject.toml`.
- [ ] Implement INT8 weight-only linear layer.
- [ ] Implement NF4 weight quantization transform.
