"""Optimizer and learning rate scheduler factory functions.

Provides `create_optimizer`, `create_scheduler`, and parameter grouping utilities
for LLM, Vision, and Segmentation models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Adagrad, Optimizer, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)

# Optimizer Registry
OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
}

# Scheduler Registry
SCHEDULERS = {
    "step": StepLR,
    "cosine": CosineAnnealingLR,
    "cosine_restart": CosineAnnealingWarmRestarts,
    "exponential": ExponentialLR,
    "plateau": ReduceLROnPlateau,
}


def get_parameter_groups(model: nn.Module, weight_decay: float = 0.0) -> List[Dict[str, Any]]:
    """Separate model parameters into weight decay and no-decay parameter groups.

    Args:
        model (nn.Module): Neural network model instance.
        weight_decay (float, default=0.0): Weight decay penalty applied to eligible weight matrices.

    Returns:
        List[Dict[str, Any]]: List of dictionary parameter groups formatted for PyTorch optimizers.
    """
    decay = []
    no_decay = []

    whitelist_weight_modules = (
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
    )

    blacklist_weight_modules = (
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.Embedding,
    )

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            if param_name.endswith("bias") or module.__class__.__name__.lower() in (
                "layernorm",
                "rmsnorm",
                "rmsnormalization",
                "batchnorm1d",
                "batchnorm2d",
                "batchnorm3d",
                "embedding",
                "absolutepositionembedding",
                "sinusoidalpositionalembedding",
            ):
                no_decay.append(param)
            elif isinstance(module, whitelist_weight_modules):
                decay.append(param)
            elif isinstance(module, blacklist_weight_modules):
                no_decay.append(param)
            else:
                decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.9,
) -> Optimizer:
    """Instantiate optimizer with weight decay parameter grouping.

    Args:
        model (nn.Module): Neural network model instance.
        optimizer_name (str, default="adamw"): Optimizer backend ("adamw", "adam", "sgd", "rmsprop", "adagrad").
        lr (float, default=3e-4): Base learning rate.
        weight_decay (float, default=0.01): Weight decay penalty value.
        momentum (float, default=0.9): Momentum coefficient (for SGD).

    Returns:
        Optimizer: Constructed PyTorch optimizer instance.
    """
    opt_name = optimizer_name.lower()

    if opt_name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    params = get_parameter_groups(model, weight_decay)

    if opt_name == "sgd":
        optimizer = SGD(
            params,
            lr=lr,
            momentum=momentum,
        )
    else:
        optimizer = OPTIMIZERS[opt_name](
            params,
            lr=lr,
        )

    return optimizer


def linear_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int | None,
) -> LambdaLR:
    """Construct a linear warmup and linear decay learning rate scheduler.

    Args:
        optimizer (Optimizer): Target PyTorch optimizer instance.
        warmup_steps (int): Number of warmup steps.
        total_steps (int | None): Total step count.

    Returns:
        LambdaLR: Constructed LambdaLR scheduler.
    """
    if total_steps is None:
        raise ValueError("total_steps must be provided for warmup schedulers")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int | None,
) -> LambdaLR:
    """Construct a linear warmup and cosine decay learning rate scheduler.

    Args:
        optimizer (Optimizer): Target PyTorch optimizer instance.
        warmup_steps (int): Number of warmup steps.
        total_steps (int | None): Total step count.

    Returns:
        LambdaLR: Constructed LambdaLR scheduler.
    """
    if total_steps is None:
        raise ValueError("total_steps must be provided for warmup schedulers")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    total_steps: int | None = None,
    warmup_steps: int = 0,
    step_size: int = 10,
    gamma: float = 0.1,
) -> LRScheduler | ReduceLROnPlateau:
    """Instantiate a learning rate scheduler from configuration settings.

    Args:
        optimizer (Optimizer): PyTorch optimizer instance.
        scheduler_name (str, default="cosine"): Scheduler type name.
        total_steps (int | None, default=None): Total steps for step/cosine schedulers.
        warmup_steps (int, default=0): Warmup step count.
        step_size (int, default=10): Step interval for StepLR or CosineAnnealingWarmRestarts.
        gamma (float, default=0.1): Decay multiplier factor.

    Returns:
        LRScheduler | ReduceLROnPlateau: Constructed learning rate scheduler instance.
    """
    sched_name = scheduler_name.lower()

    if sched_name == "linear_warmup":
        return linear_warmup_scheduler(
            optimizer,
            warmup_steps,
            total_steps,
        )

    if sched_name == "cosine_warmup":
        return cosine_warmup_scheduler(
            optimizer,
            warmup_steps,
            total_steps,
        )

    if sched_name == "cosine":
        if total_steps is None:
            raise ValueError("total_steps required for cosine scheduler")
        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
        )

    if sched_name == "cosine_restart":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=step_size,
        )

    if sched_name == "step":
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    if sched_name == "exponential":
        return ExponentialLR(
            optimizer,
            gamma=gamma,
        )

    if sched_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            factor=gamma,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")