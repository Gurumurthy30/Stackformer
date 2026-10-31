"""Distributed training package providing DDP and FSDP initialization and process group helpers.

Exposes:
    - init_distributed: Initialize process group
    - setup_ddp: Setup PyTorch DistributedDataParallel
    - wrap_model_ddp: Wrap module in DistributedDataParallel
    - distributed_sampler: Construct DistributedSampler for dataloaders
    - get_rank, get_world_size, get_local_rank: Query process group metrics
    - is_distributed, is_main_process: Process group rank predicates
    - barrier, cleanup, cleanup_distributed: Process group synchronization and destruction
"""

from .ddp import (
    barrier,
    cleanup,
    cleanup_distributed,
    distributed_sampler,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    setup_ddp,
    wrap_model_ddp,
)

__all__ = [
    "init_distributed",
    "setup_ddp",
    "wrap_model_ddp",
    "distributed_sampler",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_distributed",
    "is_main_process",
    "barrier",
    "cleanup",
    "cleanup_distributed",
]

