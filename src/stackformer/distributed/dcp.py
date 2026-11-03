# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported and generalized from torchtitan/distributed/
# for Stackformer models and arbitrary nn.Module architectures.

"""Distributed Checkpoint (DCP) utilities for PyTorch distributed training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from .utils import is_distributed_initialized

logger = logging.getLogger(__name__)


def save_dcp(
    state_dict: dict[str, Any],
    checkpoint_dir: str | Path,
    process_group: dist.ProcessGroup | None = None,
) -> Path:
    """Save distributed model and training state using torch.distributed.checkpoint.

    Args:
        state_dict: Dictionary containing model, optimizer, or training state.
        checkpoint_dir: Directory path where checkpoint shards will be stored.
        process_group: Optional distributed process group.

    Returns:
        Path: Path to the saved checkpoint directory.
    """
    ckpt_path = Path(checkpoint_dir).resolve()
    ckpt_path.mkdir(parents=True, exist_ok=True)

    dcp.save(
        state_dict=state_dict,
        checkpoint_id=str(ckpt_path),
        process_group=process_group,
    )
    logger.info(f"Saved Distributed Checkpoint (DCP) to {ckpt_path}")
    return ckpt_path


def load_dcp(
    state_dict: dict[str, Any],
    checkpoint_dir: str | Path,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
    """Load distributed checkpoint into a state dictionary in-place.

    Args:
        state_dict: Target dictionary (matching parameter shapes/structures) to load into.
        checkpoint_dir: Directory path containing the DCP checkpoint shards.
        process_group: Optional distributed process group.

    Returns:
        dict[str, Any]: The loaded state_dict.
    """
    ckpt_path = Path(checkpoint_dir).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DCP checkpoint directory not found at: {ckpt_path}")

    dcp.load(
        state_dict=state_dict,
        checkpoint_id=str(ckpt_path),
        process_group=process_group,
    )
    logger.info(f"Loaded Distributed Checkpoint (DCP) from {ckpt_path}")
    return state_dict


def dcp_to_state_dict(
    checkpoint_dir: str | Path,
    model_template: nn.Module,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, torch.Tensor]:
    """Consolidate a DCP checkpoint directory into a single un-sharded model state dictionary.

    Loads sharded parameters from the DCP checkpoint directly into the un-sharded model
    template's state dict structure.

    Args:
        checkpoint_dir: Directory path containing DCP checkpoint shards.
        model_template: Unsharded baseline nn.Module (on CPU or target device).
        process_group: Optional process group.

    Returns:
        dict[str, torch.Tensor]: Consolidated, un-sharded model state dictionary.
    """
    state_dict = {"model": model_template.state_dict()}
    load_dcp(state_dict, checkpoint_dir, process_group=process_group)
    return state_dict["model"]
