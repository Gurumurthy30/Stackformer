"""Base class and rank gating for training loggers."""

from __future__ import annotations

import abc
from typing import Any

import torch


def is_rank_zero() -> bool:
    """Return True if running on rank 0 or in a non-distributed process."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


class Logger(abc.ABC):
    """Abstract base class for all training metrics and metadata loggers."""

    def log_config(self, config_dict: dict[str, Any]) -> None:
        """Log hyperparameter configuration dictionary."""
        pass

    @abc.abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics dictionary at a specific training step."""
        pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log textual annotations or sample generations."""
        pass

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log visual figures or plots."""
        pass

    def finish(self) -> None:
        """Flush buffers and cleanly finalize logger sessions."""
        pass
