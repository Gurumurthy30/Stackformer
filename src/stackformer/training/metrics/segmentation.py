"""Semantic segmentation metrics: MeanIoU and Dice."""

from __future__ import annotations

from typing import Any
import torch

from stackformer.training.metrics.base import Metric, _extract_predictions, _extract_target


class MeanIoU(Metric):
    """Mean Intersection-over-Union (MeanIoU) metric for semantic segmentation.

    Args:
        num_classes: Total number of segmentation categories.
        ignore_index: Target index to ignore (default: -100).
    """

    name: str = "mean_iou"
    mode: str = "max"

    def __init__(self, num_classes: int, ignore_index: int = -100) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.intersection = torch.zeros(self.num_classes, dtype=torch.long)
        self.union = torch.zeros(self.num_classes, dtype=torch.long)

    def update(self, outputs: Any, batch: Any) -> None:
        preds = _extract_predictions(outputs)
        targets = _extract_target(batch).to(preds.device)

        if preds.ndim > targets.ndim:
            preds = preds.argmax(dim=1 if preds.ndim == 4 else -1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        mask = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]

        for c in range(self.num_classes):
            c_pred = preds == c
            c_target = targets == c
            self.intersection[c] += (c_pred & c_target).sum().cpu()
            self.union[c] += (c_pred | c_target).sum().cpu()

    def compute(self) -> float:
        self._sync()
        ious = []
        for c in range(self.num_classes):
            u = self.union[c].float()
            if u > 0:
                ious.append((self.intersection[c].float() / u).item())
        return sum(ious) / len(ious) if ious else 0.0


class Dice(Metric):
    """Dice similarity coefficient metric for segmentation tasks.

    Args:
        num_classes: Total number of segmentation categories.
        ignore_index: Target index to ignore (default: -100).
    """

    name: str = "dice"
    mode: str = "max"

    def __init__(self, num_classes: int, ignore_index: int = -100) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.intersection = torch.zeros(self.num_classes, dtype=torch.long)
        self.total = torch.zeros(self.num_classes, dtype=torch.long)

    def update(self, outputs: Any, batch: Any) -> None:
        preds = _extract_predictions(outputs)
        targets = _extract_target(batch).to(preds.device)

        if preds.ndim > targets.ndim:
            preds = preds.argmax(dim=1 if preds.ndim == 4 else -1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        mask = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]

        for c in range(self.num_classes):
            c_pred = preds == c
            c_target = targets == c
            self.intersection[c] += (c_pred & c_target).sum().cpu()
            self.total[c] += c_pred.sum().cpu() + c_target.sum().cpu()

    def compute(self) -> float:
        self._sync()
        dices = []
        for c in range(self.num_classes):
            tot = self.total[c].float()
            if tot > 0:
                dices.append((2.0 * self.intersection[c].float() / tot).item())
        return sum(dices) / len(dices) if dices else 0.0

