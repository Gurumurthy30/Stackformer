"""Training utility helpers."""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn


def prepare_inputs(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in batch to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, tuple | list):
        return tuple(prepare_inputs(x, device) for x in batch)
    elif isinstance(batch, dict):
        return {k: prepare_inputs(v, device) for k, v in batch.items()}
    return batch


def forward_batch(model: nn.Module, batch: Any) -> Any:
    """Execute forward pass on model unpacking inputs according to batch format."""
    if isinstance(batch, dict):
        if "input_ids" in batch and len(batch) == 2 and ("labels" in batch or "targets" in batch):
            return model(batch["input_ids"])
        elif (
            "pixel_values" in batch
            and len(batch) == 2
            and ("labels" in batch or "targets" in batch)
        ):
            return model(batch["pixel_values"])

        # Match dictionary keys with model.forward parameter names
        try:
            sig = inspect.signature(model.forward)
            model_kwargs = {k: v for k, v in batch.items() if k in sig.parameters}
            if model_kwargs:
                return model(**model_kwargs)
        except Exception:
            pass
        return model(**batch)

    elif isinstance(batch, tuple | list):
        if len(batch) >= 1:
            return model(batch[0])
        raise ValueError("Cannot forward empty batch sequence to model.")

    return model(batch)
