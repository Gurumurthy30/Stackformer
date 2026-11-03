"""Shared test utility functions for StackFormer test suite."""

import random

import numpy as np
import torch


def _checkpoint(msg: str, **kv) -> None:
    """Print a lightweight, greppable debug checkpoint during test execution.

    Output format:
        [CHECKPOINT] message | key1=val1, key2=val2
    """
    kv_str = ", ".join(f"{k}={v}" for k, v in kv.items()) if kv else ""
    suffix = f" | {kv_str}" if kv_str else ""
    print(f"[CHECKPOINT] {msg}{suffix}")


def set_seed(seed: int = 42) -> None:
    """Set random seed for Python random, numpy, and PyTorch for test determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
