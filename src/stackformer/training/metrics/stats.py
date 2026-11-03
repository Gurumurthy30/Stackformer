"""Training-time statistics and throughput helper."""

from __future__ import annotations

import time

import torch


class TrainStats:
    """Helper for tracking step time, throughput, learning rates, and memory.

    Note: This is an operational trainer statistic tracker, not an evaluation Metric subclass.
    """

    def __init__(self) -> None:
        self.step_start_time: float = time.perf_counter()
        self.last_step_time: float = 0.0

    def start_step(self) -> None:
        """Mark the beginning of an optimizer step."""
        self.step_start_time = time.perf_counter()

    def end_step(
        self,
        samples: int = 0,
        tokens: int = 0,
        loss: float | None = None,
        lr: float | None = None,
        grad_norm: float | None = None,
    ) -> dict[str, float]:
        """Mark the end of an optimizer step and compute instantaneous throughput stats."""
        now = time.perf_counter()
        elapsed = max(now - self.step_start_time, 1e-6)
        self.last_step_time = elapsed

        stats = {
            "step_time_sec": elapsed,
        }
        if samples > 0:
            stats["samples_per_sec"] = samples / elapsed
        if tokens > 0:
            stats["tokens_per_sec"] = tokens / elapsed
        if loss is not None:
            stats["train_loss"] = loss
        if lr is not None:
            stats["lr"] = lr
        if grad_norm is not None:
            stats["grad_norm"] = grad_norm

        if torch.cuda.is_available():
            stats["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["gpu_mem_max_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return stats
