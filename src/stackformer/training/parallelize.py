"""Model transformation pipeline for distributed training, activation checkpointing, and compilation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import torch.nn as nn

if TYPE_CHECKING:
    from .config import TrainConfig

logger = logging.getLogger(__name__)


def apply_model_transformations(model: nn.Module, config: TrainConfig) -> nn.Module:
    """Apply model transformations in fixed order: TP -> AC -> compile -> FSDP2.

    Order is strictly enforced:
        1. TP (Tensor Parallelism): shards linear projections / attention across ranks
        2. AC (Activation Checkpointing): wraps repeating layer blocks
        3. torch.compile: compiles blocks or full module (with TP communication hooks if active)
        4. FSDP2 (Fully Sharded Data Parallel): shards parameters across data parallel ranks

    Args:
        model: Original PyTorch nn.Module.
        config: TrainConfig containing parallelism and transformation settings.

    Returns:
        nn.Module: Transformed model.
    """
    # 1. TP (Tensor Parallelism)
    # Reserved for Step 7 / Step 9 (torch parallelize_module)

    # 2. Activation Checkpointing
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

    # 4. FSDP2 (Fully Sharded Data Parallel)
    # Hook for Step 7 (FSDP2 module wrapping)

    return model
