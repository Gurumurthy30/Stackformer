"""Accelerate-style distributed launcher supporting torchrun and programmatic multi-process spawn."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import Any

import torch.distributed as dist
import torch.multiprocessing as mp

from stackformer.distributed.utils import (
    configure_windows_environment,
    get_default_backend,
    get_free_port,
    validate_platform_and_device,
)

logger = logging.getLogger(__name__)


def _worker_wrapper(
    rank: int,
    world_size: int,
    master_port: int,
    backend: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Internal target entrypoint executed by each spawned child worker process."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Windows PyTorch wheels are compiled without libuv support.
    # Setting USE_LIBUV=0 forces the standard socket TCPStore.
    configure_windows_environment()

    init_method = f"tcp://127.0.0.1:{master_port}?use_libuv=0" if sys.platform == "win32" else None

    if init_method is not None:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
    else:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    try:
        fn(*args, **kwargs)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def launch(
    fn: Callable[..., Any],
    num_processes: int = 2,
    backend: str | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: float = 120.0,
) -> None:
    """Launch a function across multiple distributed processes.

    Supports both CLI-launched environments (via torchrun) and interactive/notebook
    programmatic spawns via torch.multiprocessing.spawn.

    Args:
        fn: Target callable to run on each worker process.
        num_processes: Number of processes to spawn when running programmatically.
        backend: Distributed backend ("nccl" or "gloo"). Defaults to "nccl" on CUDA/Linux,
            "gloo" on CPU or native Windows.
        args: Positional arguments to forward to `fn`.
        kwargs: Keyword arguments to forward to `fn`.
        timeout_seconds: Maximum spawn timeout duration before raising a TimeoutError.
    """
    kwargs = kwargs or {}

    # 1. Existing torchrun / distributed environment detection
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _ = int(os.environ["WORLD_SIZE"])
        if not dist.is_initialized():
            resolved_backend = backend or get_default_backend()
            configure_windows_environment()
            dist.init_process_group(backend=resolved_backend)
        try:
            fn(*args, **kwargs)
        finally:
            pass
        return

    # 2. Programmatic multi-process spawn
    if num_processes <= 1:
        # Single-process execution requires no process group
        fn(*args, **kwargs)
        return

    resolved_backend = backend or get_default_backend()

    # Validate platform and backend compatibility
    validate_platform_and_device(
        parallel="ddp",
        backend=resolved_backend,
        device="cuda" if resolved_backend == "nccl" else "cpu",
        world_size=num_processes,
    )

    port = get_free_port()
    logger.info(
        f"Spawning {num_processes} processes with backend='{resolved_backend}' on port {port}."
    )

    try:
        mp.spawn(
            _worker_wrapper,
            args=(num_processes, port, resolved_backend, fn, args, kwargs),
            nprocs=num_processes,
            join=True,
        )
    except mp.ProcessRaisedException as exc:
        logger.error(f"Distributed process execution failed: {exc}")
        raise exc
