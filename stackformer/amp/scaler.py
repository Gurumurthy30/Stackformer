"""Automatic mixed precision (AMP) scaler and autocast wrappers.

Provides `AMPScaler` for managing loss scaling and precision casting across PyTorch training runs.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict

import torch
import torch.nn as nn


class AMPScaler:
    """Wrapper around PyTorch CUDA/AMP GradScaler and autocast context.

    Simple explanation:
        `AMPScaler` safely manages float16/bfloat16 precision casting and loss scaling
        to prevent underflow during backpropagation on CUDA devices.

    Note:
        AMP acceleration requires CUDA support. If ``enabled=True`` but CUDA is unavailable
        (e.g., CPU-only environment), ``self.enabled`` is automatically set to ``False``.

    Constructor args:
        enabled (bool, default=True): Whether AMP is enabled for training.

    Example:
        >>> scaler = AMPScaler(enabled=True)
        >>> with scaler.autocast():
        ...     loss = model(x)
        >>> scaler.scale(loss).backward()
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled and torch.cuda.is_available())

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)
            self._autocast = lambda: torch.amp.autocast(device_type="cuda", enabled=self.enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)
            self._autocast = lambda: torch.cuda.amp.autocast(enabled=self.enabled)

    def autocast(self) -> Any:
        """Return autocast context manager if AMP is enabled, or nullcontext otherwise."""
        if not self.enabled:
            return nullcontext()
        return self._autocast()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss tensor to prevent underflow during backward pass.

        Args:
            loss (torch.Tensor): Unscaled loss tensor.

        Returns:
            torch.Tensor: Scaled loss tensor.
        """
        if not self.enabled:
            return loss
        return self.scaler.scale(loss)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer, unscaling gradients first if required.

        Args:
            optimizer (torch.optim.Optimizer): PyTorch optimizer instance.
        """
        if not self.enabled:
            optimizer.step()
            return
        self.scaler.step(optimizer)

    def update(self) -> None:
        """Update scale factor after optimizer step."""
        if self.enabled:
            self.scaler.update()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale optimizer gradients prior to clipping.

        Args:
            optimizer (torch.optim.Optimizer): PyTorch optimizer instance.
        """
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def state_dict(self) -> Dict[str, Any]:
        """Return scaler state dictionary for checkpointing.

        Returns:
            Dict[str, Any]: Scaler state dictionary.
        """
        if not self.enabled:
            return {}
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore scaler state from checkpoint dictionary.

        Args:
            state_dict (Dict[str, Any]): Saved scaler state dictionary.
        """
        if self.enabled:
            self.scaler.load_state_dict(state_dict)

    @property
    def is_enabled(self) -> bool:
        """Check if AMP scaling is currently active."""
        return self.enabled


def initialize_scaler(enabled: bool = True) -> AMPScaler:
    """Factory helper to construct an AMPScaler instance.

    Args:
        enabled (bool, default=True): Enable AMP scaling.

    Returns:
        AMPScaler: Constructed AMPScaler instance.
    """
    return AMPScaler(enabled=enabled)


def scale_loss(loss: torch.Tensor, scaler: AMPScaler | None) -> torch.Tensor:
    """Scale loss tensor using optional AMPScaler instance.

    Args:
        loss (torch.Tensor): Loss tensor.
        scaler (AMPScaler | None): Active AMPScaler instance or None.

    Returns:
        torch.Tensor: Scaled or unscaled loss tensor.
    """
    if scaler is None:
        return loss
    return scaler.scale(loss)


def step_optimizer(optimizer: torch.optim.Optimizer, scaler: AMPScaler | None) -> None:
    """Step optimizer using optional AMPScaler instance.

    Args:
        optimizer (torch.optim.Optimizer): Target optimizer.
        scaler (AMPScaler | None): Active AMPScaler instance or None.
    """
    if scaler is None:
        optimizer.step()
        return
    scaler.step(optimizer)


def update_scaler(scaler: AMPScaler | None) -> None:
    """Update dynamic scale factor on optional AMPScaler instance.

    Args:
        scaler (AMPScaler | None): Active AMPScaler instance or None.
    """
    if scaler is not None:
        scaler.update()

