# Multimodal Models Subsystem

Status: Planned, not implemented

## Purpose
Provides vision-language models (e.g., CLIP, LLaVA, PaLI) and document AI architectures combining visual encoders with autoregressive language model decoders.

## Planned Contents
- Vision-Language cross-modal projection layers (`Linear`, `MLP`, `PerceiverResampler`).
- Joint encoder-decoder multimodal architectures.
- Document AI layout and text embedding models.

## Checklist
- [ ] Define multimodal model configuration schema extending `ModelConfig`.
- [ ] Implement multimodal connector and projection modules.
- [ ] Build end-to-end vision-language architecture.
- [ ] Integrate with `Trainer` and multimodal dataset collation.
