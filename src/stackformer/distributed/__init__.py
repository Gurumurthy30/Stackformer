# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stackformer distributed primitives and device mesh."""

from .mesh import MeshAxisName, ParallelDims, unfold_dp_axes
from .hardware import device_type, get_device_info, get_local_device

__all__ = [
    "MeshAxisName",
    "ParallelDims",
    "unfold_dp_axes",
    "device_type",
    "get_device_info",
    "get_local_device",
]
