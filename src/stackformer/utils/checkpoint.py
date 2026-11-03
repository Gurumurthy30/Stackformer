"""Checkpoint saving and loading utilities supporting PyTorch and SafeTensors formats.

Provides helpers to serialize and deserialize model weights with automatic format
detection (.pt, .pth, .bin, .safetensors) and safe dictionary unpacking.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_model_weights(model: nn.Module, path: str | Path) -> None:
    """Save model weights to disk in PyTorch (.pt/.pth) or SafeTensors format.

    Args:
        model (nn.Module): Model whose state dictionary will be serialized.
        path (str | Path): Destination file path. If ending with '.safetensors',
            weights are saved using SafeTensors format; otherwise PyTorch format is used.

    Example:
        >>> from stackformer.utils.checkpoint import save_model_weights
        >>> save_model_weights(model, "checkpoints/model.safetensors")
        >>> save_model_weights(model, "checkpoints/model.pt")
    """
    path_obj = Path(path)
    if path_obj.parent:
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    if path_obj.suffix.lower() == ".safetensors":
        try:
            import safetensors.torch

            safetensors.torch.save_model(model, str(path_obj))
        except ImportError as e:
            raise ImportError(
                "safetensors is required to save .safetensors files. Install via `pip install safetensors`."
            ) from e
    else:
        state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
        torch.save(state_dict, str(path_obj))


def load_model_weights(
    model: nn.Module,
    path: str | Path,
    strict: bool = True,
    device: torch.device | str = "cpu",
) -> None:
    """Load model weights from disk into a PyTorch model.

    Supports SafeTensors (.safetensors) and PyTorch (.pt, .pth, .bin) checkpoint files,
    automatically unwrapping nested 'state_dict' or 'model' keys if present.

    Args:
        model (nn.Module): Target model to receive weights.
        path (str | Path): Path to the checkpoint file.
        strict (bool, default=True): Whether to strictly enforce key matching.
        device (torch.device | str, default="cpu"): Compute device for deserialization.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ImportError: If loading .safetensors without safetensors package installed.

    Example:
        >>> from stackformer.utils.checkpoint import load_model_weights
        >>> load_model_weights(model, "checkpoints/model.safetensors")
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path_obj}")

    if path_obj.suffix.lower() == ".safetensors":
        try:
            import safetensors.torch

            safetensors.torch.load_model(model, str(path_obj), strict=strict, device=str(device))
        except ImportError as e:
            raise ImportError(
                "safetensors is required to load .safetensors files. Install via `pip install safetensors`."
            ) from e
    else:
        loaded = torch.load(str(path_obj), map_location=device, weights_only=True)
        if isinstance(loaded, dict):
            if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                state_dict = loaded["state_dict"]
            elif "model" in loaded and isinstance(loaded["model"], dict):
                state_dict = loaded["model"]
            else:
                state_dict = loaded
        else:
            state_dict = loaded

        model.load_state_dict(state_dict, strict=strict)
