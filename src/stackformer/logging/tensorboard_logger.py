"""TensorBoard logger backend for training metric visualization.

Provides `TensorBoardLogger` class using PyTorch SummaryWriter.
"""

from __future__ import annotations

import datetime
import os
from typing import Dict

from stackformer.utils.utils import is_main_process


class TensorBoardLogger:
    """Writes scalar training metrics to TensorBoard event logs.

    Constructor args:
        log_dir (str, default="logs"): Parent directory path for log outputs.
        experiment_name (str, default="stackformer_run"): Experiment name identifier.
        auto_timestamp (bool, default=True): Append timestamp to experiment log directory name.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "stackformer_run",
        auto_timestamp: bool = True,
    ) -> None:
        self.enabled = is_main_process()
        self.step = 0
        self.writer = None

        if not self.enabled:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "tensorboard is not installed. Install with `pip install tensorboard`."
            ) from exc

        if auto_timestamp:
            experiment_name = (
                f"{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def log(self, metrics: Dict[str, float]) -> None:
        """Add scalar metrics to TensorBoard summary writer.

        Args:
            metrics (Dict[str, float]): Key-value metric dictionary.
        """
        if not self.enabled or not metrics or self.writer is None:
            return

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, self.step)
        self.step += 1

    def flush(self) -> None:
        """Flush pending TensorBoard events to disk."""
        if self.enabled and self.writer:
            self.writer.flush()

    def close(self) -> None:
        """Flush and close active TensorBoard summary writer."""
        if self.enabled and self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
