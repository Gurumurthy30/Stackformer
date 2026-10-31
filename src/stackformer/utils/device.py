"""Device detection, memory formatting, and tensor placement utility functions.

Provides `get_device`, `move_to_device`, `print_device_info`, and CUDA cache management helpers.
"""

from __future__ import annotations

from typing import Any

import torch

from stackformer.utils.utils import is_main_process, print_once


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve target compute device based on availability and explicit user preference.

    Priority order:
      1. Explicit `device` argument when provided.
      2. CUDA device if available.
      3. MPS device (Apple Silicon) if available.
      4. CPU fallback.

    Args:
        device (str | torch.device | None, default=None): Preferred target device or "auto".

    Returns:
        torch.device: Canonical PyTorch device instance.
    """
    if device is not None and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(obj: Any, device: str | torch.device) -> Any:
    """Recursively move tensors, dictionaries, lists, or tuples to a target compute device.

    Args:
        obj (Any): Input tensor or nested container object.
        device (str | torch.device): Destination compute device.

    Returns:
        Any: Object with all constituent tensors moved to `device`.
    """
    if torch.is_tensor(obj):
        target_device = torch.device(device)
        return obj.to(target_device, non_blocking=(target_device.type == "cuda"))
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    return obj


def format_memory(bytes_value: int) -> str:
    """Format integer bytes value as a human-readable gigabyte string (GB).

    Args:
        bytes_value (int): Byte count.

    Returns:
        str: Formatted string (e.g. "16.00 GB").
    """
    return f"{bytes_value / (1024 ** 3):.2f} GB"


def print_device_info() -> None:
    """Print runtime compute device specification and memory details on rank 0."""
    if not is_main_process():
        return

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print_once(f"Device: CUDA:{idx} {props.name}")
        print_once(f"GPU Memory: {format_memory(props.total_memory)}")
        return

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print_once("Device: MPS")
        return

    print_once("Device: CPU")


def clear_cuda_cache() -> None:
    """Empty the CUDA memory cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize() -> None:
    """Synchronize current CUDA device operations if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

