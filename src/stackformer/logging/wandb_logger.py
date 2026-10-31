"""Weights & Biases (W&B) logger backend for experiment tracking.

Provides `WandBLogger` class to log scalar metrics and watch model parameters.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch.nn as nn

from stackformer.utils.utils import is_main_process


class WandBLogger:
    """Writes scalar metrics and model parameter gradients to Weights & Biases (W&B).

    Constructor args:
        project (str, default="stackformer"): W&B project name.
        experiment_name (str | None, default=None): Name identifier for experiment run.
        config (dict | None, default=None): Hyperparameter dictionary to record.
        entity (str | None, default=None): W&B entity / team space name.
        watch_model (bool, default=False): If True, track model parameter gradients via `wandb.watch`.
    """

    def __init__(
        self,
        project: str = "stackformer",
        experiment_name: Optional[str] = None,
        config: Optional[dict] = None,
        entity: Optional[str] = None,
        watch_model: bool = False,
    ) -> None:
        self.enabled = is_main_process()
        self.run = None
        self.step = 0
        self.start_time = time.time()
        self.watch_model_enabled = watch_model

        if not self.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install with `pip install wandb`."
            ) from exc

        self.wandb = wandb
        self.run = self.wandb.init(
            project=project, name=experiment_name, entity=entity, config=config
        )

    def log(self, metrics: Dict[str, float]) -> None:
        """Log key-value metrics to W&B run.

        Args:
            metrics (Dict[str, float]): Key-value metric dictionary.
        """
        if not self.enabled or not metrics:
            return

        clean_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        clean_metrics["step"] = self.step
        clean_metrics["time"] = time.time() - self.start_time

        try:
            self.wandb.log(clean_metrics, step=self.step)
        except Exception:
            pass

        self.step += 1

    def log_model(self, model: nn.Module) -> None:
        """Attach W&B gradient and parameter tracking to a PyTorch model.

        Args:
            model (nn.Module): Target PyTorch model module.
        """
        if not self.enabled or not self.watch_model_enabled:
            return
        try:
            self.wandb.watch(model)
        except Exception:
            pass

    def finish(self) -> None:
        """Safely finish W&B run and sync remaining logs."""
        if not self.enabled or self.run is None:
            return
        try:
            self.wandb.finish()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.finish()
        except Exception:
            pass

