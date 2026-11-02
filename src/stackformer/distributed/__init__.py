# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stackformer distributed primitives, device mesh, activation checkpointing, and compile."""

from .activation_checkpoint import apply_activation_checkpointing
from .compile import apply_compile, is_compile_available
from .hardware import device_type, get_device_info, get_local_device
from .mesh import MeshAxisName, ParallelDims, unfold_dp_axes

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
]
