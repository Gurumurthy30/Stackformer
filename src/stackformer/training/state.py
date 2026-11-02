"""Training state tracking for Stackformer Trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainState:
    """Tracks dynamic training progress, step counters, and best metrics.

    Attributes:
        epoch: Current epoch index (0-indexed).
        global_step: Number of completed optimizer steps across all epochs.
        step_in_epoch: Number of batches processed in the current epoch.
        total_micro_batches: Total micro-batches processed.
        best_metric: Best value of the monitored metric so far.
        best_epoch: Epoch in which the best metric occurred.
        best_step: Global step in which the best metric occurred.
        should_stop: Flag signaling training should terminate.
    """

    epoch: int = 0
    global_step: int = 0
    step_in_epoch: int = 0
    total_micro_batches: int = 0
    best_metric: float | None = None
    best_epoch: int = 0
    best_step: int = 0
    should_stop: bool = False

    def is_better(self, current: float, mode: str = "min") -> bool:
        """Check if current metric value improves on best_metric."""
        if self.best_metric is None:
            return True
        if mode == "min":
            return current < self.best_metric
        elif mode == "max":
            return current > self.best_metric
        raise ValueError(f"Unknown mode '{mode}', expected 'min' or 'max'.")

    def update_best(self, current: float, mode: str = "min") -> bool:
        """Update best metric if current value improves upon it. Returns True if improved."""
        if self.is_better(current, mode):
            self.best_metric = current
            self.best_epoch = self.epoch
            self.best_step = self.global_step
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a dictionary."""
        return asdict(self)

    def load_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state from a dictionary."""
        for key, val in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, val)

