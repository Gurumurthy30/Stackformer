"""Base class and lifecycle hooks for training callbacks."""

from __future__ import annotations

from typing import Any


class Callback:
    """Abstract base class for training loop lifecycle listeners.

    Subclasses can override any hook method to execute custom actions, log data,
    or request early termination by setting `trainer.state.should_stop = True`.
    """

    def on_train_start(self, trainer: Any) -> None:
        """Called at the beginning of `trainer.train()` before the first epoch."""
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each training epoch."""
        pass

    def on_step_end(self, trainer: Any, step: int, metrics: dict[str, float]) -> None:
        """Called after completing an optimizer step."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each training epoch."""
        pass

    def on_eval_end(self, trainer: Any, metrics: dict[str, float]) -> None:
        """Called immediately after evaluation completes."""
        pass

    def on_checkpoint_save(self, trainer: Any, checkpoint_dir: str) -> None:
        """Called after a checkpoint has been successfully saved to disk."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training, even if aborted early."""
        pass

