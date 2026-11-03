# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported and generalized from torchtitan/distributed/
# for Stackformer models and arbitrary nn.Module architectures.

"""Distributed Data Parallel (DDP) wrapper and hooks for Stackformer models."""

from __future__ import annotations

import logging
from typing import Any
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .utils import is_distributed_initialized

logger = logging.getLogger(__name__)


def apply_ddp(
    model: nn.Module,
    device: torch.device | str = "cpu",
    process_group: dist.ProcessGroup | None = None,
    find_unused_parameters: bool = False,
    bucket_cap_mb: int = 25,
) -> nn.Module:
    """Wrap a model with PyTorch DistributedDataParallel if distributed training is active.

    If training in a single-process environment, returns the model unchanged.

    Args:
        model: PyTorch module to wrap.
        device: Device where the model parameters reside ("cpu" or "cuda:...").
        process_group: Process group for gradient synchronization (default: default group).
        find_unused_parameters: Traverse autograd graph to identify unused parameters.
        bucket_cap_mb: Size of all-reduce communication buckets in megabytes.

    Returns:
        nn.Module: DDP-wrapped model or original model if world_size == 1.
    """
    if not is_distributed_initialized() or dist.get_world_size() == 1:
        return model

    if isinstance(model, DDP):
        return model

    dev = torch.device(device)
    device_ids = [dev.index] if (dev.type == "cuda" and dev.index is not None) else None

    ddp_model = DDP(
        model,
        device_ids=device_ids,
        process_group=process_group,
        find_unused_parameters=find_unused_parameters,
        bucket_cap_mb=bucket_cap_mb,
    )

    logger.info(
        f"Applied DDP across world_size={dist.get_world_size()} ranks (rank={dist.get_rank()})."
    )
    return ddp_model
