"""Training state container for tracking execution progress.

Provides `TrainingState` to manage epoch counts, global step counters, device placement,
and distributed environment metadata.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from stackformer.utils.utils import get_rank, get_world_size


class TrainingState:
    """Container for mutable training runtime state and component references.

    Simple explanation:
        `TrainingState` bundles model, optimizer, scheduler, scaler, device info,
        and progress metrics (epoch, global step) into a single object passed
        across training loops and checkpointing managers.

    Constructor args:
        model (nn.Module | None, default=None): Target model instance.
        optimizer (Optimizer | None, default=None): Optimizer instance.
        scheduler (LRScheduler | None, default=None): Learning rate scheduler.
        scaler (torch.cuda.amp.GradScaler | None, default=None): AMP gradient scaler.
        device (torch.device | str | None, default=None): Target compute device.
        config (dict | None, default=None): Training configuration dictionary.

    Example:
        >>> state = TrainingState(model=model, optimizer=optimizer, device="cuda")
        >>> state.step()
        >>> state.global_step
        1
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler: Any | None = None,
        device: torch.device | str | None = None,
        config: Dict[str, Any] | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        self.epoch = 0
        self.global_step = 0
        self.batch_idx = 0
        self.config = config or {}

        if self.model is not None:
            self.model.to(self.device)

    def step(self) -> None:
        """Increment the global optimization step counter by 1."""
        self.global_step += 1

    def next_epoch(self) -> None:
        """Increment current epoch count by 1 and reset batch index."""
        self.epoch += 1
        self.batch_idx = 0

    def reset_batch(self) -> None:
        """Reset intra-epoch batch index back to 0."""
        self.batch_idx = 0

    def get_lr(self) -> float | None:
        """Retrieve current learning rate from the active optimizer param group.

        Returns:
            float | None: Active learning rate, or None if no optimizer is attached.
        """
        if self.optimizer is None:
            return None
        return float(self.optimizer.param_groups[0]["lr"])

    def to_device(self, obj: Any) -> Any:
        """Recursively move tensors, dictionaries, or lists to the state device.

        Args:
            obj (Any): Target object or container.

        Returns:
            Any: Object with nested tensors placed on `self.device`.
        """
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(x) for x in obj)
        return obj

    def metadata(self) -> Dict[str, int]:
        """Return progress metrics dictionary suitable for checkpoint serialization.

        Returns:
            Dict[str, int]: Dictionary with epoch, global_step, and batch_idx.
        """
        return {"epoch": self.epoch, "global_step": self.global_step, "batch_idx": self.batch_idx}

    def load_metadata(self, metadata: Dict[str, int]) -> None:
        """Restore progress metrics from a saved metadata dictionary.

        Args:
            metadata (Dict[str, int]): Dictionary containing epoch, global_step, and batch_idx.
        """
        self.epoch = metadata.get("epoch", 0)
        self.global_step = metadata.get("global_step", 0)
        self.batch_idx = metadata.get("batch_idx", 0)

    @property
    def amp_enabled(self) -> bool:
        """Check if AMP gradient scaling is enabled."""
        return bool(self.scaler is not None and getattr(self.scaler, "is_enabled", False))

    @property
    def rank(self) -> int:
        """Return process rank in distributed process group."""
        return get_rank()

    @property
    def world_size(self) -> int:
        """Return total world size in distributed process group."""
        return get_world_size()

    @property
    def is_distributed(self) -> bool:
        """Check if executing in a distributed environment (world_size > 1)."""
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """Check if current process is rank 0 main process."""
        return self.rank == 0

