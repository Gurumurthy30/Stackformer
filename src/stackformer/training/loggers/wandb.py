"""Weights & Biases (Wandb) metrics logger with lazy imports."""

from __future__ import annotations

from typing import Any, Optional

from stackformer.training.loggers.base import Logger, is_rank_zero


class WandbLogger(Logger):
    """Logs training metrics, hyperparameters, and artifacts to Weights & Biases (wandb).

    Raises an informative ImportError upon instantiation if `wandb` is not installed.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import wandb

            self._wandb = wandb
        except ImportError as e:
            raise ImportError("wandb is required for WandbLogger. Install it with: pip install wandb") from e

        self.run = None
        if is_rank_zero():
            self.run = self._wandb.init(project=project, name=name, config=config, **kwargs)

    def log_config(self, config_dict: dict[str, Any]) -> None:
        if self.run is not None and is_rank_zero():
            self.run.config.update(config_dict, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.run is not None and is_rank_zero():
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None and is_rank_zero():
            self.run.finish()
            self.run = None
