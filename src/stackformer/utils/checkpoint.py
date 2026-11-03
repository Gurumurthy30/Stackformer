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

    # Unwrap any DDP or compile wrapper
    unwrapped = model
    while hasattr(unwrapped, "_orig_mod") or hasattr(unwrapped, "module"):
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
        elif hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

    if path_obj.suffix.lower() == ".safetensors":
        try:
            import safetensors.torch

            safetensors.torch.save_model(unwrapped, str(path_obj))
        except ImportError as e:
            raise ImportError(
                "safetensors is required to save .safetensors files. Install via `pip install safetensors`."
            ) from e
    else:
        state_dict = {k: v.contiguous() for k, v in unwrapped.state_dict().items()}
        torch.save(state_dict, str(path_obj))


def load_weights(
    model: nn.Module,
    path: str | Path,
    strict: bool = True,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load model weights from a .safetensors file, .pt file, or checkpoint directory.

    Automatically unwraps directory checkpoints (locating `model.safetensors` or `model.pt`),
    strips `module.` (DDP) and `_orig_mod.` (torch.compile) prefixes, and loads state
    into the given model.

    Args:
        model (nn.Module): Target model to receive weights.
        path (str | Path): Path to a checkpoint file or directory.
        strict (bool, default=True): Whether to strictly enforce key matching.
        device (torch.device | str, default="cpu"): Compute device for deserialization.

    Returns:
        nn.Module: The target model instance with updated weights.

    Raises:
        FileNotFoundError: If checkpoint file or directory does not exist or contains no weights.
        ImportError: If loading .safetensors without safetensors package installed.
        RuntimeError: If strict=True and state_dict keys do not match target model.

    Example:
        >>> import stackformer
        >>> stackformer.load_weights(model, "checkpoints/step_00000100")
        >>> stackformer.load_weights(model, "checkpoints/model.safetensors")
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path_obj}")

    if path_obj.is_dir():
        st_file = path_obj / "model.safetensors"
        pt_file = path_obj / "model.pt"
        if st_file.exists():
            path_obj = st_file
        elif pt_file.exists():
            path_obj = pt_file
        else:
            raise FileNotFoundError(
                f"No model.safetensors or model.pt found in checkpoint directory: {path_obj}"
            )

    if path_obj.suffix.lower() == ".safetensors":
        try:
            import safetensors.torch

            unwrapped = model
            while hasattr(unwrapped, "_orig_mod") or hasattr(unwrapped, "module"):
                if hasattr(unwrapped, "_orig_mod"):
                    unwrapped = unwrapped._orig_mod
                elif hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module

            try:
                safetensors.torch.load_model(
                    unwrapped, str(path_obj), strict=strict, device=str(device)
                )
                return model
            except Exception:
                # Fallback to load_file if keys have wrapper prefixes or need custom handling
                state_dict = safetensors.torch.load_file(str(path_obj), device=str(device))
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

    # Clean keys: recursively strip _orig_mod. and module. prefixes
    cleaned_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        while "_orig_mod." in clean_k:
            clean_k = clean_k.replace("_orig_mod.", "")
        while clean_k.startswith("module."):
            clean_k = clean_k[7:]
        cleaned_dict[clean_k] = v

    model.load_state_dict(cleaned_dict, strict=strict)
    return model


load_model_weights = load_weights
