"""Reusable training, evaluation, and prediction loop execution primitives.

Provides `train_epoch`, `eval_epoch`, and `predict_loop` functions.
"""

from __future__ import annotations

from typing import Any, List

import torch
from tqdm import tqdm

from stackformer.utils.utils import is_main_process


def train_epoch(engine: Any, dataloader: Any, epoch: int) -> None:
    """Execute a single training epoch loop over the given dataloader.

    Args:
        engine (Any): Engine instance executing steps and tracking state.
        dataloader (Any): PyTorch DataLoader providing training batches.
        epoch (int): Current epoch number index.
    """
    if dataloader is None:
        raise ValueError("train dataloader must not be None")

    state = engine.state
    state.model.train()

    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    iterator = enumerate(dataloader)
    show_progress = not engine.live_monitoring and is_main_process()
    if show_progress:
        iterator = tqdm(iterator, total=len(dataloader), desc=f"Train Epoch {epoch}")

    for _, batch in iterator:
        if engine.reached_max_train_steps():
            break

        metrics = engine._train_step(batch)
        if show_progress and metrics:
            display_metrics = {}
            if "loss" in metrics:
                display_metrics["loss"] = f"{metrics['loss']:.4f}"
            if "lr" in metrics and metrics["lr"] is not None:
                display_metrics["lr"] = f"{metrics['lr']:.2e}"
            iterator.set_postfix(display_metrics)


def eval_epoch(engine: Any, dataloader: Any, epoch: int) -> None:
    """Execute a non-optimizing evaluation epoch loop over the given dataloader.

    Args:
        engine (Any): Engine instance executing validation steps.
        dataloader (Any): PyTorch DataLoader providing validation batches.
        epoch (int): Current epoch number index.
    """
    if dataloader is None:
        raise ValueError("validation dataloader must not be None")

    state = engine.state
    state.model.eval()

    iterator = enumerate(dataloader)
    show_progress = not engine.live_monitoring and is_main_process()
    if show_progress:
        iterator = tqdm(iterator, total=len(dataloader), desc=f"Validation Epoch {epoch}")

    max_eval_steps = getattr(engine, "max_eval_steps", None)
    with torch.no_grad():
        for batch_idx, batch in iterator:
            if max_eval_steps is not None and batch_idx >= int(max_eval_steps):
                break
            metrics = engine._eval_step(batch)
            if show_progress and metrics and "val_loss" in metrics:
                iterator.set_postfix({"val_loss": f"{metrics['val_loss']:.4f}"})


def predict_loop(engine: Any, dataloader: Any) -> List[Any]:
    """Execute forward passes in eval mode over a dataloader and accumulate raw model outputs.

    Args:
        engine (Any): Engine instance providing model and device placement.
        dataloader (Any): DataLoader providing prediction inputs.

    Returns:
        List[Any]: List of raw model predictions for each batch.
    """
    outputs = []
    engine.state.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = engine._prepare_batch(batch)
            outputs.append(engine._forward_model(inputs))
    return outputs


# Backwards compatibility aliases
train_loop = train_epoch
validation_loop = eval_epoch

