"""Evaluation loop and metric accumulation helper."""

from __future__ import annotations

from typing import Any

import torch

from .utils import forward_batch, prepare_inputs


def run_evaluation(trainer: Any) -> dict[str, float]:
    """Execute evaluation loop over val_loader, compute metrics, and handle best-checkpointing."""
    if trainer.val_loader is None:
        return {}

    trainer.model.eval()
    total_loss = 0.0
    num_batches = 0
    max_eval = trainer.config.max_eval_steps

    for m in trainer.metrics:
        m.reset()

    with torch.no_grad():
        for batch in trainer.val_loader:
            if max_eval is not None and num_batches >= max_eval:
                break
            batch = prepare_inputs(batch, trainer.device)
            with trainer.precision.autocast():
                outputs = forward_batch(trainer.model, batch)
                loss = trainer.loss_fn(outputs, batch)

            total_loss += loss.item()
            num_batches += 1

            for m in trainer.metrics:
                m.update(outputs, batch)

    avg_loss = total_loss / max(1, num_batches)
    eval_metrics = {"val_loss": avg_loss}

    for m in trainer.metrics:
        res = m.compute()
        if isinstance(res, dict):
            for k, v in res.items():
                eval_metrics[f"val_{k}"] = v
        else:
            eval_metrics[f"val_{m.name}"] = res

    # Callbacks and loggers dispatch
    for cb in trainer.callbacks:
        cb.on_eval_end(trainer, eval_metrics)
    for lg in trainer.loggers:
        lg.log_metrics(eval_metrics, step=trainer.state.global_step)

    # Best-model checkpointing trigger
    if trainer.config.save_best and trainer.config.monitor in eval_metrics:
        mon_val = eval_metrics[trainer.config.monitor]
        if trainer.state.update_best(mon_val, mode=trainer.config.mode):
            trainer.save_best_model(eval_metrics)

    return eval_metrics
