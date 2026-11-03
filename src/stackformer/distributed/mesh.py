# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported and generalized from torchtitan/distributed/mesh.py
# for Stackformer data-parallel device mesh operations.

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from stackformer.distributed.hardware import device_type

logger = logging.getLogger(__name__)

__all__ = [
    "MeshAxisName",
    "ParallelDims",
    "unfold_dp_axes",
    "unfold_dp_axis",
]


class MeshAxisName(StrEnum):
    """Names for data-parallel axes of a ``DeviceMesh``."""

    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"


def unfold_dp_axis(axis: MeshAxisName | str) -> tuple[MeshAxisName, ...]:
    """Expand logical ``dp`` into concrete dense storage mesh axes."""
    axis_name = MeshAxisName(axis)
    if axis_name == MeshAxisName.DP:
        return (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD)
    return (axis_name,)


def unfold_dp_axes(axes: Iterable[MeshAxisName | str]) -> list[str]:
    """Expand logical ``dp`` into concrete dense storage mesh axes."""
    return [concrete_axis.value for axis in axes for concrete_axis in unfold_dp_axis(axis)]


@dataclass
class ParallelDims:
    """Data-parallel dimension coordinator for Stackformer training."""

    dp_replicate: int = 1
    dp_shard: int = -1
    world_size: int = 1
    _single_axis_meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _multi_axis_meshes: dict[tuple[str, ...], DeviceMesh] = field(default_factory=dict)
    _world_mesh: DeviceMesh | None = None

    @classmethod
    def from_config(cls, parallelism_config: Any, world_size: int) -> ParallelDims:
        return cls(
            dp_replicate=getattr(parallelism_config, "data_parallel_replicate_degree", 1),
            dp_shard=getattr(parallelism_config, "data_parallel_shard_degree", -1),
            world_size=world_size,
        )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        dp_replicate, dp_shard = self.dp_replicate, self.dp_shard
        assert dp_replicate >= 1, f"dp_replicate must be >= 1, got {dp_replicate}"
        assert dp_shard == -1 or dp_shard >= 1, f"dp_shard must be -1 or >= 1, got {dp_shard}"
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // dp_replicate
        assert dp_shard >= 1

        assert dp_replicate * dp_shard == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) != WORLD_SIZE({self.world_size})"
        )

    def _mesh_exist(self, name: str, degree: int) -> bool:
        if name == "dp_shard":
            # Keep the DP storage axis alive at size 1 so fully_shard can install MixedPrecisionPolicy
            return True
        return degree > 1

    def build_mesh(self) -> DeviceMesh:
        """Build the device mesh with data-parallel dimensions."""
        logger.info(
            f"Building data-parallel device mesh: dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, world_size={self.world_size}"
        )

        self._world_mesh = init_device_mesh(device_type, (self.world_size,), mesh_dim_names=("world",))

        if self.dp_replicate > 1 and self.dp_shard > 1:
            full_dp_mesh = self._world_mesh._unflatten(
                0,
                (self.dp_replicate, self.dp_shard),
                ("dp_replicate", "dp_shard"),
            )
            self._single_axis_meshes = {
                "dp_replicate": full_dp_mesh["dp_replicate"],
                "dp_shard": full_dp_mesh["dp_shard"],
                "dp": self._world_mesh,
            }
            self._global_meshes = {"dp": full_dp_mesh}
        else:
            self._single_axis_meshes = {
                "dp_replicate": self._world_mesh,
                "dp_shard": self._world_mesh,
                "dp": self._world_mesh,
            }
            self._global_meshes = {"dp": self._world_mesh}

        return self._world_mesh

    def get_optional_mesh(
        self,
        dims: str | list[str],
        *,
        include_singleton_axes: bool = False,
    ) -> DeviceMesh | None:
        """Get a device mesh by dimension name(s), returning None if not enabled."""
        if not self._single_axis_meshes:
            self.build_mesh()

        if isinstance(dims, str):
            dims = [dims]

        for mesh_name in dims:
            if mesh_name not in self._single_axis_meshes:
                raise ValueError(
                    f"Invalid mesh dim: '{mesh_name}'. Valid dimensions are: {list(self._single_axis_meshes.keys())}"
                )

        if not include_singleton_axes and any(
            not self._mesh_exist(dim, self._single_axis_meshes[dim].size()) for dim in dims
        ):
            return None

        if len(dims) == 1:
            return self._single_axis_meshes[dims[0]]

        key = tuple(dims)
        if key in self._multi_axis_meshes:
            return self._multi_axis_meshes[key]

        candidates = [
            (name, global_mesh)
            for name, global_mesh in self._global_meshes.items()
            if global_mesh.mesh_dim_names is not None and set(dims).issubset(set(global_mesh.mesh_dim_names))
        ]
        if not candidates:
            raise ValueError(f"Invalid mesh name combinations {dims}.")
        submesh = candidates[0][1][key]
        self._multi_axis_meshes[key] = submesh
        return submesh

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        """Get a device mesh by dimension name(s), raising if not available."""
        mesh = self.get_optional_mesh(dims)
        if mesh is None:
            enabled_str = "enabled (size > 1)" if isinstance(dims, str) else "all enabled"
            raise ValueError(
                f"Mesh '{dims}' is not available. Ensure the corresponding parallelism dimension is {enabled_str}."
            )
        return mesh

    def get_activated_mesh(self, axes: list[str]) -> DeviceMesh | None:
        """Submesh of axes filtered to those actually enabled in this run."""
        if not self._single_axis_meshes:
            self.build_mesh()
        axes = [axis for axis in axes if axis in self._single_axis_meshes and self.get_optional_mesh(axis) is not None]
        return self.get_optional_mesh(axes) if axes else None

    def get_all_one_dimensional_meshes(self) -> dict[str, DeviceMesh]:
        """Get all enabled one-dimensional device meshes."""
        if not self._single_axis_meshes:
            self.build_mesh()
        return {
            k: v
            for k, v in self._single_axis_meshes.items()
            if v.ndim == 1 and v.size() > 1 and self._mesh_exist(k, v.size())
        }

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    @property
    def fsdp_enabled(self) -> bool:
        return self.dp_shard_enabled
