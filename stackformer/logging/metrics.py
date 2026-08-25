"""Metric calculation utilities and streaming metric tracking.

Provides functions for accuracy, perplexity, precision, recall, and F1 score, along with
`MetricTracker` for tracking running statistics during model training.
"""

from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Dict

import torch
import torch.distributed as dist


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def distributed_mean(value: float | torch.Tensor) -> float:
    """Compute mean of a scalar metric value across distributed process ranks.

    Args:
        value (float | torch.Tensor): Scalar value to reduce across processes.

    Returns:
        float: Distributed average across process group ranks.
    """
    tensor = torch.as_tensor(float(value), dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy between predictions and targets.

    Args:
        preds (torch.Tensor): Logits or prediction tensor.
        targets (torch.Tensor): Target labels tensor.

    Returns:
        float: Accuracy score in range [0.0, 1.0].
    """
    pred_ids = preds.argmax(dim=-1) if preds.ndim > targets.ndim else preds
    return float((pred_ids == targets).float().mean().item())


def perplexity(loss: float | torch.Tensor) -> float:
    """Compute language modeling perplexity from cross-entropy loss.

    Args:
        loss (float | torch.Tensor): Cross-entropy loss value.

    Returns:
        float: Perplexity exp(loss).
    """
    value = float(loss.item() if torch.is_tensor(loss) else loss)
    try:
        return float(math.exp(value))
    except OverflowError:
        return float("inf")


def precision_recall_f1(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    """Compute binary classification precision, recall, and F1 score.

    Note:
        Assumes binary target labels (0 or 1). Multi-class targets are not supported.

    Args:
        preds (torch.Tensor): Predictions tensor.
        targets (torch.Tensor): Binary ground truth targets tensor.

    Returns:
        tuple[float, float, float]: Tuple of (precision, recall, f1_score).
    """
    pred_ids = preds.argmax(dim=-1) if preds.ndim > targets.ndim else preds
    pred_ids = pred_ids.view(-1)
    targets = targets.view(-1)

    tp = ((pred_ids == 1) & (targets == 1)).sum().item()
    fp = ((pred_ids == 1) & (targets == 0)).sum().item()
    fn = ((pred_ids == 0) & (targets == 1)).sum().item()

    precision_val = _safe_div(tp, tp + fp)
    recall_val = _safe_div(tp, tp + fn)
    f1_val = _safe_div(2 * precision_val * recall_val, precision_val + recall_val)
    return float(precision_val), float(recall_val), float(f1_val)


def precision(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute binary classification precision."""
    return precision_recall_f1(preds, targets)[0]


def recall(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute binary classification recall."""
    return precision_recall_f1(preds, targets)[1]


def f1_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute binary classification F1 score."""
    return precision_recall_f1(preds, targets)[2]


class _Metric:
    """Internal running sum and count accumulator for a single named metric."""

    def __init__(self) -> None:
        self.reset()

    def update(self, value: float) -> None:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return
        self.last = val
        self.sum += val
        self.count += 1

    def avg(self) -> float:
        return 0.0 if self.count == 0 else self.sum / self.count

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.last = None


class MetricTracker:
    """Tracks running averages and step timings for training metrics.

    Example:
        >>> tracker = MetricTracker()
        >>> tracker.update("loss", 0.5)
        >>> tracker.avg("loss")
        0.5
    """

    def __init__(self) -> None:
        self.metrics = defaultdict(_Metric)
        self.start_time = time.time()
        self.step_start_time: float | None = None

    def update(self, name: str, value: float) -> None:
        """Update named metric with a new scalar observation."""
        self.metrics[name].update(value)

    def avg(self, name: str) -> float | None:
        """Get running average value for named metric."""
        metric = self.metrics.get(name)
        return None if metric is None else metric.avg()

    def compute(self, name: str) -> float | None:
        """Alias for avg(name)."""
        return self.avg(name)

    def reset(self) -> None:
        """Reset all tracked metrics and restart epoch timer."""
        for metric in self.metrics.values():
            metric.reset()
        self.start_time = time.time()

    def get_all(self, reduce_distributed: bool = False) -> Dict[str, float]:
        """Return dictionary of all tracked average metric values."""
        results = {name: metric.avg() for name, metric in self.metrics.items()}
        if reduce_distributed:
            return {name: distributed_mean(value) for name, value in results.items()}
        return results

    def start_step_timer(self) -> None:
        """Start iteration step timer."""
        self.step_start_time = time.time()

    def end_step_timer(self) -> float | None:
        """Stop step timer, update 'step_time' metric, and return step duration in seconds."""
        if self.step_start_time is None:
            return None
        step_time = time.time() - self.step_start_time
        self.update("step_time", step_time)
        self.step_start_time = None
        return step_time

    def update_tokens(self, tokens: int) -> None:
        """Update total tokens processed and calculate tokens_per_sec throughput."""
        self.update("tokens", tokens)
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            tokens_per_sec = self.metrics["tokens"].sum / elapsed
            self.update("tokens_per_sec", tokens_per_sec)

    def update_perplexity(self, loss: float | torch.Tensor) -> None:
        """Update perplexity metric from cross-entropy loss."""
        self.update("perplexity", perplexity(loss))


# Backward-compatible PascalCase alias (Issue 26)
metrics = MetricTracker
Metrics = MetricTracker

