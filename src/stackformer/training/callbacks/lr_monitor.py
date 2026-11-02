"""Learning rate monitor callback for tracking scheduled rates."""

from __future__ import annotations

from typing import Any

from stackformer.training.callbacks.base import Callback


class LRMonitor(Callback):
    """Monitors and logs optimizer learning rates at every optimizer step."""

    def on_step_end(self, trainer: Any, step: int, metrics: dict[str, float]) -> None:
        if trainer.optimizer is None:
            return

        lrs = [group["lr"] for group in trainer.optimizer.param_groups if "lr" in group]
        if not lrs:
            return

        lr_metrics = {"lr": lrs[0]}
        if len(lrs) > 1:
            for idx, lr in enumerate(lrs):
                lr_metrics[f"lr/group_{idx}"] = lr

        for logger in trainer.loggers:
            logger.log_metrics(lr_metrics, step=step)

