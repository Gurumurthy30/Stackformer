"""Perplexity metric for autoregressive language models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from stackformer.training.metrics.base import Metric, _extract_predictions, _extract_target


class Perplexity(Metric):
    """Perplexity metric for language models.

    Computes exponentiated cross-entropy loss over token predictions:
        ``Perplexity = exp(sum(loss) / total_tokens)``

    Adheres to the same next-token shift logic and `loss_mode` options as `cross_entropy_loss`.

    Args:
        ignore_index: Label index to ignore (default: -100).
        loss_mode: Shift strategy ("auto", "lm", "classification").
    """

    name: str = "perplexity"
    mode: str = "min"

    def __init__(self, ignore_index: int = -100, loss_mode: str = "auto") -> None:
        self.ignore_index = ignore_index
        self.loss_mode = loss_mode.lower()
        self.reset()

    def reset(self) -> None:
        self.total_loss = torch.tensor(0.0, dtype=torch.float64)
        self.total_tokens = torch.tensor(0, dtype=torch.long)

    def update(self, outputs: Any, batch: Any) -> None:
        logits = _extract_predictions(outputs)
        targets = _extract_target(batch).to(logits.device)

        if self.loss_mode == "lm" or (
            self.loss_mode == "auto" and logits.ndim == 3 and targets.ndim == 2
        ):
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=self.ignore_index,
                reduction="sum",
            )
            mask = shift_targets != self.ignore_index
            tokens = mask.sum()
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.ignore_index,
                reduction="sum",
            )
            mask = targets != self.ignore_index
            tokens = mask.sum()

        if tokens > 0:
            self.total_loss += loss.detach().cpu().to(torch.float64)
            self.total_tokens += tokens.detach().cpu()

    def compute(self) -> float:
        self._sync()
        if self.total_tokens.item() == 0:
            return 0.0
        mean_loss = self.total_loss.float() / self.total_tokens.float()
        return torch.exp(mean_loss).item()
