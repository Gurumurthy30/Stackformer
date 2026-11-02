"""Classification evaluation metrics: Accuracy, TopKAccuracy, Precision, Recall, and F1."""

from __future__ import annotations

from typing import Any
import torch

from stackformer.training.metrics.base import Metric, _extract_predictions, _extract_target


class Accuracy(Metric):
    """Classification accuracy metric over discrete class predictions.

    Args:
        ignore_index: Target label index to exclude from calculation (default: -100).
    """

    name: str = "accuracy"
    mode: str = "max"

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.correct = torch.tensor(0, dtype=torch.long)
        self.total = torch.tensor(0, dtype=torch.long)

    def update(self, outputs: Any, batch: Any) -> None:
        preds = _extract_predictions(outputs)
        targets = _extract_target(batch).to(preds.device)

        if preds.ndim > targets.ndim:
            preds = preds.argmax(dim=-1)

        mask = targets != self.ignore_index
        if mask.any():
            self.correct += (preds[mask] == targets[mask]).sum().cpu()
            self.total += mask.sum().cpu()

    def compute(self) -> float:
        self._sync()
        if self.total.item() == 0:
            return 0.0
        return (self.correct.float() / self.total.float()).item()


class TopKAccuracy(Metric):
    """Top-K classification accuracy for categorical prediction distributions.

    Args:
        k: Number of highest probability candidates to consider (default: 5).
        ignore_index: Target label index to exclude from calculation (default: -100).
    """

    name: str = "topk_accuracy"
    mode: str = "max"

    def __init__(self, k: int = 5, ignore_index: int = -100) -> None:
        self.k = k
        self.ignore_index = ignore_index
        self.name = f"top{k}_accuracy"
        self.reset()

    def reset(self) -> None:
        self.correct = torch.tensor(0, dtype=torch.long)
        self.total = torch.tensor(0, dtype=torch.long)

    def update(self, outputs: Any, batch: Any) -> None:
        preds = _extract_predictions(outputs)
        targets = _extract_target(batch).to(preds.device)

        k = min(self.k, preds.size(-1))
        topk_indices = torch.topk(preds, k=k, dim=-1).indices
        targets_expanded = targets.unsqueeze(-1)

        match = (topk_indices == targets_expanded).any(dim=-1)
        mask = targets != self.ignore_index
        if mask.any():
            self.correct += match[mask].sum().cpu()
            self.total += mask.sum().cpu()

    def compute(self) -> float:
        self._sync()
        if self.total.item() == 0:
            return 0.0
        return (self.correct.float() / self.total.float()).item()


class Precision(Metric):
    """Precision metric across discrete categories with macro or micro averaging.

    Args:
        num_classes: Number of distinct prediction classes.
        average: Averaging strategy, 'macro' or 'micro' (default: 'macro').
        ignore_index: Target label index to exclude (default: -100).
    """

    name: str = "precision"
    mode: str = "max"

    def __init__(self, num_classes: int, average: str = "macro", ignore_index: int = -100) -> None:
        self.num_classes = num_classes
        self.average = average.lower()
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.tp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fp = torch.zeros(self.num_classes, dtype=torch.long)

    def update(self, outputs: Any, batch: Any) -> None:
        preds = _extract_predictions(outputs)
        targets = _extract_target(batch).to(preds.device)

        if preds.ndim > targets.ndim:
            preds = preds.argmax(dim=-1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        mask = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]

        for c in range(self.num_classes):
            c_pred = preds == c
            c_target = targets == c
            self.tp[c] += (c_pred & c_target).sum().cpu()
            self.fp[c] += (c_pred & ~c_target).sum().cpu()

    def compute(self) -> float:
        self._sync()
        if self.average == "micro":
            total_tp = self.tp.sum().float()
            total_fp = self.fp.sum().float()
            denom = total_tp + total_fp
            return (total_tp / denom).item() if denom > 0 else 0.0

        # macro
        precisions = []
        for c in range(self.num_classes):
            denom = self.tp[c].float() + self.fp[c].float()
            if denom > 0:
                precisions.append((self.tp[c].float() / denom).item())
        return sum(precisions) / len(precisions) if precisions else 0.0


class Recall(Metric):
    """Recall metric across discrete categories with macro or micro averaging.

    Args:
        num_classes: Number of distinct prediction classes.
        average: Averaging strategy, 'macro' or 'micro' (default: 'macro').
        ignore_index: Target label index to exclude (default: -100).
    """

    name: str = "recall"
    mode: str = "max"

    def __init__(self, num_classes: int, average: str = "macro", ignore_index: int = -100) -> None:
        self.num_classes = num_classes
        self.average = average.lower()
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.tp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fn = torch.zeros(self.num_classes, dtype=torch.long)

    def update(self, outputs: Any, batch: Any) -> None:
        preds = _extract_predictions(outputs)
        targets = _extract_target(batch).to(preds.device)

        if preds.ndim > targets.ndim:
            preds = preds.argmax(dim=-1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        mask = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]

        for c in range(self.num_classes):
            c_pred = preds == c
            c_target = targets == c
            self.tp[c] += (c_pred & c_target).sum().cpu()
            self.fn[c] += (~c_pred & c_target).sum().cpu()

    def compute(self) -> float:
        self._sync()
        if self.average == "micro":
            total_tp = self.tp.sum().float()
            total_fn = self.fn.sum().float()
            denom = total_tp + total_fn
            return (total_tp / denom).item() if denom > 0 else 0.0

        # macro
        recalls = []
        for c in range(self.num_classes):
            denom = self.tp[c].float() + self.fn[c].float()
            if denom > 0:
                recalls.append((self.tp[c].float() / denom).item())
        return sum(recalls) / len(recalls) if recalls else 0.0


class F1(Metric):
    """F1 Score metric combining precision and recall.

    Args:
        num_classes: Number of distinct prediction classes.
        average: Averaging strategy, 'macro' or 'micro' (default: 'macro').
        ignore_index: Target label index to exclude (default: -100).
    """

    name: str = "f1"
    mode: str = "max"

    def __init__(self, num_classes: int, average: str = "macro", ignore_index: int = -100) -> None:
        self.num_classes = num_classes
        self.average = average.lower()
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.tp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fn = torch.zeros(self.num_classes, dtype=torch.long)

    def update(self, outputs: Any, batch: Any) -> None:
        preds = _extract_predictions(outputs)
        targets = _extract_target(batch).to(preds.device)

        if preds.ndim > targets.ndim:
            preds = preds.argmax(dim=-1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        mask = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]

        for c in range(self.num_classes):
            c_pred = preds == c
            c_target = targets == c
            self.tp[c] += (c_pred & c_target).sum().cpu()
            self.fp[c] += (c_pred & ~c_target).sum().cpu()
            self.fn[c] += (~c_pred & c_target).sum().cpu()

    def compute(self) -> float:
        self._sync()
        if self.average == "micro":
            tp = self.tp.sum().float()
            fp = self.fp.sum().float()
            fn = self.fn.sum().float()
            p = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
            r = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
            denom = p + r
            return (2 * p * r / denom).item() if denom > 0 else 0.0

        f1s = []
        for c in range(self.num_classes):
            tp = self.tp[c].float()
            fp = self.fp[c].float()
            fn = self.fn[c].float()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if (p + r) > 0:
                f1s.append(2 * p * r / (p + r))
        return sum(f1s) / len(f1s) if f1s else 0.0

