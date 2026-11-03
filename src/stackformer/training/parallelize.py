"""Model transformation pipeline for distributed training, activation checkpointing, and compilation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import torch.nn as nn

if TYPE_CHECKING:
    from .config import TrainConfig

logger = logging.getLogger(__name__)


def apply_model_transformations(model: nn.Module, config: TrainConfig) -> nn.Module:
    """Apply model transformations in fixed order: AC -> compile -> DDP/FSDP2.

    Order is strictly enforced:
        1. AC (Activation Checkpointing): wraps repeating layer blocks
        2. torch.compile: compiles blocks or full module
        3. DDP / FSDP2 (Data Parallelism): wraps with DDP or shards parameters across FSDP2 ranks

    Args:
        model: Original PyTorch nn.Module.
        config: TrainConfig containing parallelism and transformation settings.

    Returns:
        nn.Module: Transformed model.
    """
    # 1. Activation Checkpointing
    if getattr(config, "activation_checkpointing", "none") != "none":
        from stackformer.distributed.activation_checkpoint import apply_activation_checkpointing

        model = apply_activation_checkpointing(
            model,
            mode=config.activation_checkpointing,
        )

    # 3. torch.compile
    if getattr(config, "compile", False):
        from stackformer.distributed.compile import apply_compile

        model = apply_compile(
            model,
            compile_mode=config.compile_mode,
            per_block=True,
        )

    # 4. FSDP2 / DDP (Data Parallelism)
    parallel = getattr(config, "parallel", "none").lower().strip()
    if parallel == "ddp":
        from stackformer.distributed.ddp import apply_ddp

        model = apply_ddp(model, device=config.resolve_device())
    elif parallel == "fsdp2":
        from stackformer.distributed.fsdp import apply_fsdp

        model = apply_fsdp(
            model,
            mixed_precision=config.mixed_precision,
            device=config.resolve_device(),
        )

    return model

