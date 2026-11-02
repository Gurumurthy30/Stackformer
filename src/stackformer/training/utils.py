"""Training utility helpers."""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn


def prepare_inputs(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in batch to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (tuple, list)):
        return tuple(prepare_inputs(x, device) for x in batch)
    elif isinstance(batch, dict):
        return {k: prepare_inputs(v, device) for k, v in batch.items()}
    return batch


def forward_batch(model: nn.Module, batch: Any) -> Any:
    """Execute forward pass on model unpacking inputs according to batch format."""
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return model(batch[0])
    elif isinstance(batch, dict):
        if "input_ids" in batch:
            return model(batch["input_ids"])
        elif "pixel_values" in batch:
            return model(batch["pixel_values"])
        return model(**batch)
    return model(batch)

