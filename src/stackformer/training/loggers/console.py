"""Console logger for human-readable terminal output."""

from __future__ import annotations

import logging
from typing import Any
from stackformer.training.loggers.base import Logger, is_rank_zero

logger = logging.getLogger("stackformer.training")


class ConsoleLogger(Logger):
    """Outputs training metrics to the console or python logging cleanly without duplicates."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def log_config(self, config_dict: dict[str, Any]) -> None:
        if not is_rank_zero() or not self.verbose:
            return
        logger.info("Training configuration initialized.")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not is_rank_zero() or not self.verbose:
            return
        parts = [f"step {step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        msg = " | ".join(parts)
        # Avoid duplicating tqdm lines if running inside an active progress bar
        # by using module logger at INFO
        logger.info(msg)
