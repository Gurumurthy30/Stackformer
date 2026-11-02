"""TensorBoard metrics logger with lazy imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from stackformer.training.loggers.base import Logger, is_rank_zero


class TensorBoardLogger(Logger):
    """Logs training metrics and graphs to TensorBoard event files.

    Raises an informative ImportError upon instantiation if `tensorboard` is not installed.
    """

    def __init__(self, log_dir: Optional[str | Path] = None, **kwargs: Any) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "tensorboard is required for TensorBoardLogger. "
                "Install it with: pip install tensorboard"
            ) from e

        self.log_dir = Path(log_dir or "./checkpoints/logs/tensorboard").resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), **kwargs) if is_rank_zero() else None

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.writer is None or not is_rank_zero():
            return
        for tag, val in metrics.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(tag, val, global_step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        if self.writer is None or not is_rank_zero():
            return
        self.writer.add_text(tag, text, global_step=step)

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

