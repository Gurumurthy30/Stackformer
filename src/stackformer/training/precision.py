"""Mixed precision execution and GradScaler management."""

from __future__ import annotations

import contextlib
from typing import Any
import torch


class PrecisionManager:
    """Manages AMP autocast context and gradient scaling.

    Args:
        mixed_precision: "no", "fp16", or "bf16".
        device: Target execution torch.device.
    """

    def __init__(self, mixed_precision: str = "no", device: torch.device | None = None) -> None:
        self.mixed_precision = mixed_precision.lower().strip()
        self.device = device or torch.device("cpu")
        self.device_type = self.device.type

        self.enabled = self.mixed_precision in {"fp16", "bf16"}
        if self.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        elif self.mixed_precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # GradScaler is only needed and supported for fp16 on CUDA
        self.scaler: torch.amp.GradScaler | None = None
        if self.mixed_precision == "fp16" and self.device_type == "cuda":
            self.scaler = torch.amp.GradScaler(self.device_type, enabled=True)

    @contextlib.contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if not self.enabled:
            yield
            return

        # Handle device_type for torch.autocast
        dev = self.device_type
        if dev not in {"cuda", "cpu", "xpu", "mps", "hpu"}:
            dev = "cpu"

        # On CPU, only bfloat16 is supported for autocast
        if dev == "cpu" and self.dtype == torch.float16:
            yield
            return

        with torch.autocast(device_type=dev, dtype=self.dtype, enabled=self.enabled):
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss tensor if a GradScaler is active."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step the optimizer using scaler if active, else regular step."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def state_dict(self) -> dict[str, Any]:
        """Return scaler state dictionary."""
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load scaler state dictionary."""
        if self.scaler is not None and state_dict:
            self.scaler.load_state_dict(state_dict)

