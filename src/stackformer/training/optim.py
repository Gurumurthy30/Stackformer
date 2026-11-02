"""Optimizer and learning rate scheduler factories for Stackformer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

if TYPE_CHECKING:
    from .config import TrainConfig


def build_optimizer(model: nn.Module, config: TrainConfig) -> Optimizer:
    """Build optimizer with weight decay applied to 2D+ weights only.

    Biases and 1D normalization weights (LayerNorm, RMSNorm) are excluded from weight decay.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not any(nd in name for nd in ["bias", "norm", "ln_"]):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": config.weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    if not param_groups:
        param_groups = [{"params": model.parameters(), "weight_decay": config.weight_decay}]

    opt_name = config.optimizer.lower().strip()
    if opt_name == "adamw":
        return AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95), eps=1e-8)
    elif opt_name == "adam":
        return Adam(param_groups, lr=config.lr, betas=(0.9, 0.999), eps=1e-8)
    elif opt_name == "sgd":
        return SGD(param_groups, lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer '{config.optimizer}', expected 'adamw', 'adam', or 'sgd'.")


def build_scheduler(
    optimizer: Optimizer, config: TrainConfig, total_steps: int
) -> LRScheduler | None:
    """Build learning rate scheduler with optional warmup and decay."""
    sched_name = config.scheduler.lower().strip()
    if sched_name in {"none", ""}:
        return None

    warmup_steps = config.warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(decay_steps)
        progress = min(1.0, max(0.0, progress))
        if sched_name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif sched_name == "linear":
            return max(0.0, 1.0 - progress)
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

