# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stackformer distributed primitives, device mesh, activation checkpointing, DDP, FSDP2, and DCP."""

from .activation_checkpoint import apply_activation_checkpointing
from .compile import apply_compile, is_compile_available
from .dcp import dcp_to_state_dict, load_dcp, save_dcp
from .ddp import apply_ddp
from .fsdp import apply_fsdp, build_fsdp_policies
from .hardware import device_type, get_device_info, get_local_device
from .mesh import MeshAxisName, ParallelDims, unfold_dp_axes
from .utils import (
    configure_windows_environment,
    dist_max,
    dist_mean,
    dist_sum,
    get_default_backend,
    get_free_port,
    get_rank,
    get_world_size,
    is_distributed_initialized,
    is_rank_zero,
    validate_platform_and_device,
)

__all__ = [
    "MeshAxisName",
    "ParallelDims",
    "unfold_dp_axes",
    "device_type",
    "get_device_info",
    "get_local_device",
    "apply_activation_checkpointing",
    "apply_compile",
    "is_compile_available",
    "apply_ddp",
    "apply_fsdp",
    "build_fsdp_policies",
    "save_dcp",
    "load_dcp",
    "dcp_to_state_dict",
    "configure_windows_environment",
    "dist_max",
    "dist_mean",
    "dist_sum",
    "get_default_backend",
    "get_free_port",
    "get_rank",
    "get_world_size",
    "is_rank_zero",
    "validate_platform_and_device",
]
