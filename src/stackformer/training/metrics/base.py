"""Base class and utilities for vectorized evaluation metrics."""

from __future__ import annotations

import abc
from typing import Any
import torch


def _extract_target(batch: Any) -> torch.Tensor:
    """Helper to extract target tensor from a batch."""
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2:
            return batch[1]
        return batch[0]
    elif isinstance(batch, dict):
        for key in ["labels", "targets", "target", "label", "input_ids"]:
            if key in batch:
                return batch[key]
    elif isinstance(batch, torch.Tensor):
        return batch
    raise ValueError(f"Could not extract target tensor from batch of type {type(batch).__name__}")


def _extract_predictions(outputs: Any) -> torch.Tensor:
    """Helper to extract predictions tensor from model outputs."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if hasattr(outputs, "logits"):
        return outputs.logits
    raise ValueError(f"Could not extract prediction tensor from output of type {type(outputs).__name__}")


class Metric(abc.ABC):
    """Abstract base class for all training and evaluation metrics.

    Metrics maintain internal accumulation state as PyTorch tensors for vectorization
    and straightforward distributed synchronization.

    Attributes:
        name: Identifier name for the metric.
        mode: Direction of metric optimization ("max" or "min").
    """

    name: str = "metric"
    mode: str = "max"

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal accumulator state."""
        pass

    @abc.abstractmethod
    def update(self, outputs: Any, batch: Any) -> None:
        """Update metric state with predictions and ground-truth batch."""
        pass

    @abc.abstractmethod
    def compute(self) -> float | dict[str, float]:
        """Compute the final scalar metric value or dictionary of metric values."""
        pass

    def _sync(self) -> None:
        """Synchronize metric tensor accumulators across distributed workers."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for attr_name, val in self.__dict__.items():
                if isinstance(val, torch.Tensor):
                    torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)

