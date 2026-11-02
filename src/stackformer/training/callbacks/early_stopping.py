"""Early stopping callback based on evaluation metric convergence."""

from __future__ import annotations

import logging
from typing import Any

from stackformer.training.callbacks.base import Callback

logger = logging.getLogger("stackformer.training")


class EarlyStopping(Callback):
    """Monitors an evaluation metric and terminates training when improvement plateaus.

    Args:
        monitor: Metric name to evaluate (default: "val_loss").
        mode: Direction of optimization, 'min' or 'max' (default: 'min').
        patience: Number of consecutive non-improving evaluation runs to tolerate (default: 3).
        min_delta: Minimum change in metric required to qualify as an improvement (default: 0.0).
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 3,
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode.lower()
        self.patience = max(1, patience)
        self.min_delta = abs(min_delta)

        self.wait_count: int = 0
        self.best_value: float | None = None
        self.stopped_epoch: int = 0

    def on_eval_end(self, trainer: Any, metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]
        if self.best_value is None:
            self.best_value = current
            self.wait_count = 0
            return

        if self.mode == "min":
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                logger.info(
                    f"Early stopping triggered! '{self.monitor}' did not improve for {self.wait_count} evaluations."
                )
                trainer.state.should_stop = True

