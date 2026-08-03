"""DistributedDataParallel (DDP) initialization and process group management helpers.

Provides functions for process group setup, rank queries, DDP model wrapping, and DistributedSampler construction.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_local_rank() -> int:
    """Return local GPU process rank from `LOCAL_RANK` environment variable."""
    return int(os.environ.get("LOCAL_RANK", "0"))


from stackformer.utils.utils import get_rank, get_world_size, is_main_process


def is_distributed() -> bool:
    """Check whether torch.distributed is initialized and active."""
    return dist.is_available() and dist.is_initialized()


def init_distributed(backend: str | None = None) -> bool:
    """Initialize `torch.distributed` process group if environment WORLD_SIZE > 1.

    Args:
        backend (str | None, default=None): Distributed backend ("nccl" for GPU, "gloo" for CPU).

    Returns:
        bool: True if process group is initialized and active, False otherwise.
    """
    if not dist.is_available() or dist.is_initialized():
        return is_distributed()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = get_local_rank()

    if world_size <= 1:
        return False

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True


def wrap_model_ddp(model: nn.Module) -> nn.Module:
    """Wrap a PyTorch model with DistributedDataParallel if running in a distributed environment.

    Args:
        model (nn.Module): Model instance to wrap.

    Returns:
        nn.Module: DDP-wrapped model instance, or original model if not distributed.
    """
    if not is_distributed():
        return model

    if torch.cuda.is_available():
        local_rank = get_local_rank()
        return DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    return DDP(model)


def distributed_sampler(dataset: Any, shuffle: bool = True) -> Any | None:
    """Construct a PyTorch `DistributedSampler` for multi-GPU data loading.

    Args:
        dataset (Any): Target dataset instance.
        shuffle (bool, default=True): Enable random index shuffling per epoch.

    Returns:
        Any | None: DistributedSampler instance if distributed, else None.
    """
    if not is_distributed():
        return None
    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=shuffle)


def barrier() -> None:
    """Synchronize all processes in the distributed process group using a barrier."""
    if is_distributed():
        dist.barrier()


def cleanup_distributed() -> None:
    """Destroy active PyTorch distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


def cleanup() -> None:
    """Alias for cleanup_distributed()."""
    cleanup_distributed()


def setup_ddp(backend: str | None = None) -> bool:
    """Alias for init_distributed(backend)."""
    return init_distributed(backend=backend)

