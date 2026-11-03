# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported and generalized from torchtitan/distributed/utils.py
# for Stackformer distributed training, platform guarding, and collectives.

from __future__ import annotations

import logging
import os
import socket
import sys

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def is_distributed_initialized() -> bool:
    """Return whether torch.distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return current distributed rank, or 0 if single process."""
    if is_distributed_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return total distributed world size, or 1 if single process."""
    if is_distributed_initialized():
        return dist.get_world_size()
    return 1


def is_rank_zero() -> bool:
    """Return True if current process is rank 0 or uninitialized single-process."""
    return get_rank() == 0


def get_free_port() -> int:
    """Find and return an ephemeral open TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def configure_windows_environment() -> None:
    """Configure environment variables on native Windows.

    Windows PyTorch wheels are compiled without libuv support. PyTorch 2.6 defaults
    TCPStore to use_libuv=True, which crashes with:
        RuntimeError: use_libuv was requested but PyTorch was build without libuv support.
    Setting USE_LIBUV=0 forces the standard socket-based TCPStore on Windows.
    """
    if sys.platform == "win32":
        if "USE_LIBUV" not in os.environ:
            os.environ["USE_LIBUV"] = "0"


def get_default_backend(device: torch.device | str | None = None) -> str:
    """Determine the optimal communication backend for the target device and OS.

    Returns:
        str: "nccl" on CUDA/accelerator under Linux/WSL2; "gloo" on CPU or native Windows.
    """
    if device is not None:
        dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dev.type == "cuda" and sys.platform != "win32" and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def validate_platform_and_device(
    parallel: str = "none",
    backend: str | None = None,
    device: torch.device | str = "cpu",
    world_size: int = 1,
) -> None:
    """Validate platform, device, backend, and parallelism strategy compatibility.

    Raises clear, actionable errors when unsupported configurations are requested.
    """
    parallel = (parallel or "none").lower().strip()
    dev = torch.device(device)
    is_windows = sys.platform == "win32"

    # 1. Windows + NCCL check
    if is_windows and backend is not None and backend.lower() == "nccl":
        raise RuntimeError(
            "Native Windows does not support the NCCL distributed backend. "
            "Multi-GPU distributed training with NCCL requires Linux or WSL2."
        )

    # 2. Windows + Multi-GPU CUDA check
    if is_windows and dev.type == "cuda" and world_size > 1:
        raise RuntimeError(
            "Native Windows does not support multi-GPU distributed training with NCCL. "
            "Please run multi-GPU training on Linux or WSL2."
        )

    # 3. FSDP2 on CPU check (PyTorch 2.6 requires device stream events)
    if parallel == "fsdp2" and dev.type == "cpu":
        raise RuntimeError(
            "FSDP2 in PyTorch 2.6 requires CUDA/accelerator streams; "
            "use parallel='ddp' for multi-process CPU training, or run FSDP2 on Linux/WSL2 with CUDA."
        )


def dist_mean(tensor: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Compute the all-reduce average of a tensor across ranks (or return self if single-process)."""
    if not is_distributed_initialized():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.AVG, group=group)
    return t


def dist_sum(tensor: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Compute the all-reduce sum of a tensor across ranks (or return self if single-process)."""
    if not is_distributed_initialized():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return t


def dist_max(tensor: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Compute the all-reduce max of a tensor across ranks (or return self if single-process)."""
    if not is_distributed_initialized():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t
