"""Callback, logger, and lifecycle event dispatch helpers."""

from __future__ import annotations

from typing import Any


def dispatch_train_start(trainer: Any) -> None:
    """Notify callbacks and log initial model configuration to loggers."""
    for cb in trainer.callbacks:
        cb.on_train_start(trainer)
    cfg_dict = trainer.config.to_dict()
    cfg_dict["num_parameters"] = sum(p.numel() for p in trainer.model.parameters())
    for lg in trainer.loggers:
        lg.log_config(cfg_dict)


def dispatch_epoch_start(trainer: Any, epoch: int) -> None:
    """Notify callbacks that an epoch has started."""
    for cb in trainer.callbacks:
        cb.on_epoch_start(trainer, epoch)


def dispatch_step_end(trainer: Any, step: int, metrics: dict[str, float]) -> None:
    """Notify callbacks and log step-level training metrics."""
    for cb in trainer.callbacks:
        cb.on_step_end(trainer, step, metrics)
    for lg in trainer.loggers:
        lg.log_metrics(metrics, step=step)


def dispatch_epoch_end(trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
    """Notify callbacks that an epoch has ended."""
    for cb in trainer.callbacks:
        cb.on_epoch_end(trainer, epoch, metrics)


def dispatch_checkpoint_save(trainer: Any, path: str) -> None:
    """Notify callbacks that a checkpoint has been saved."""
    for cb in trainer.callbacks:
        cb.on_checkpoint_save(trainer, path)


def dispatch_train_end(trainer: Any) -> None:
    """Safely notify callbacks that training has completed."""
    for cb in trainer.callbacks:
        try:
            cb.on_train_end(trainer)
        except Exception:
            pass


def close_loggers(trainer: Any) -> None:
    """Safely flush and finalize all registered loggers."""
    for lg in trainer.loggers:
        try:
            lg.finish()
        except Exception:
            pass

