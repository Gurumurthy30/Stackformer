"""General utilities for seed setting, directory creation, parameter counting, and distributed rank checks.

Provides `seed_everything`, `ensure_dir`, `count_parameters`, `timestamp`, `get_rank`, `get_world_size`, `is_main_process`, and `print_once`.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import torch
import torch.nn as nn

try:
    import numpy as np
except ImportError:  # optional dependency in minimal environments
    np = None


def seed_everything(seed: int = 42) -> None:
    """Seed supported random number generators (Python, NumPy, PyTorch) for execution reproducibility.

    Args:
        seed (int, default=42): Random seed integer value.
    """
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_dir(path: str) -> str:
    """Ensure directory path exists, creating intermediate directories if needed.

    Args:
        path (str): Target directory filepath.

    Returns:
        str: Absolute or input directory path string.
    """
    os.makedirs(path, exist_ok=True)
    return path


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Calculate total or trainable parameter count of a PyTorch neural network module.

    Args:
        model (nn.Module): Target PyTorch module.
        trainable_only (bool, default=True): Only count parameters requiring gradients.

    Returns:
        int: Total parameter count integer.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def timestamp() -> str:
    """Return current Unix timestamp string in seconds."""
    return str(int(time.time()))


def get_rank() -> int:
    """Return process rank in distributed environment (0 if non-distributed)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    """Return total process count in distributed environment (1 if non-distributed)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process() -> bool:
    """Check if current process is rank 0 main process."""
    return get_rank() == 0


def print_once(*args: Any, **kwargs: Any) -> None:
    """Print message to stdout only on global rank 0 process."""
    if is_main_process():
        print(*args, **kwargs)

