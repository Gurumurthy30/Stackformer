"""Parameter-Efficient Fine-Tuning (PEFT) subsystem for Stackformer.

Planned adapters:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA with 4-bit / 8-bit base weights)
- Prefix Tuning & Prompt Tuning
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


class PeftConfig:
    """Base configuration container for parameter-efficient fine-tuning."""

    def __init__(self, target_modules: list[str] | None = None) -> None:
        self.target_modules = target_modules or ["q_proj", "v_proj"]


class LoraConfig(PeftConfig):
    """Configuration container for LoRA (Low-Rank Adaptation)."""

    def __init__(
        self,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> None:
        super().__init__(target_modules=target_modules)
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


def get_peft_model(model: nn.Module, config: PeftConfig) -> nn.Module:
    """Wrap a base model with trainable adapter layers according to config."""
    raise NotImplementedError(
        "PEFT adapter injection is planned for Phase 3. "
        "Use full fine-tuning with Stackformer Trainer in current release."
    )


__all__ = [
    "LoraConfig",
    "PeftConfig",
    "get_peft_model",
]
