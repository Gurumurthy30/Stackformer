# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported and generalized from torchtitan/distributed/fsdp.py
# for Stackformer models and arbitrary nn.Module architectures.

from __future__ import annotations

import logging
from typing import Any
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
)

from .utils import is_distributed_initialized

logger = logging.getLogger(__name__)


def build_fsdp_policies(
    mixed_precision: str = "no",
    cpu_offload: bool = False,
) -> tuple[MixedPrecisionPolicy, OffloadPolicy]:
    """Construct PyTorch 2.6 FSDP2 mixed precision and parameter offload policies.

    Args:
        mixed_precision: Precision strategy ("no", "fp16", "bf16").
        cpu_offload: Whether to offload unsharded parameters to host memory.

    Returns:
        tuple: (MixedPrecisionPolicy, OffloadPolicy)
    """
    mixed_precision = mixed_precision.lower().strip()
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None

    if mixed_precision == "bf16":
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32
    elif mixed_precision == "fp16":
        param_dtype = torch.float16
        reduce_dtype = torch.float32

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=None,
        cast_forward_inputs=True,
    )

    offload_policy: OffloadPolicy = (
        CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy()
    )

    return mp_policy, offload_policy


def apply_fsdp(
    model: nn.Module,
    mesh: DeviceMesh | None = None,
    mixed_precision: str = "no",
    cpu_offload: bool = False,
    reshard_after_forward: bool = True,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Apply PyTorch 2.6 FSDP2 (fully_shard) across model repeating blocks and root module.

    Applied in the distributed initialization order: TP -> AC -> compile -> FSDP2.
    Shards repeating blocks exposed via `model.get_sharding_blocks()`, followed by the root module.

    Args:
        model: PyTorch nn.Module architecture.
        mesh: 1D or 2D DeviceMesh partition for data parallelism.
        mixed_precision: Mixed precision mode ("no", "fp16", "bf16").
        cpu_offload: Whether to offload parameters to CPU host memory.
        reshard_after_forward: Whether to immediately discard unsharded parameters after forward.
        device: Compute device ("cpu" or "cuda").

    Returns:
        nn.Module: FSDP2 sharded model.

    Raises:
        RuntimeError: If device is "cpu" (PyTorch 2.6 FSDP2 requires device stream events).
    """
    dev = torch.device(device)
    if dev.type == "cpu":
        raise RuntimeError(
            "FSDP2 in PyTorch 2.6 requires CUDA/accelerator streams; "
            "use parallel='ddp' for multi-process CPU training, or run FSDP2 on Linux/WSL2 with CUDA."
        )

    if not is_distributed_initialized() or dist.get_world_size() == 1:
        return model

    # 1. Retrieve blocks via Stackformer protocol or auto-detection
    if hasattr(model, "get_sharding_blocks") and callable(model.get_sharding_blocks):
        blocks = model.get_sharding_blocks()
    else:
        blocks = []
        for mod in model.modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 1:
                types = {type(c) for c in mod}
                if len(types) == 1:
                    blocks = list(mod)
                    break

    mp_policy, offload_policy = build_fsdp_policies(
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
    )

    # 2. fully_shard each repeating layer block
    for block in blocks:
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
        )

    # 3. fully_shard the root module to capture outside parameters (embeddings, heads, norms)
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=reshard_after_forward,
    )

    logger.info(
        f"Applied FSDP2 fully_shard across {len(blocks)} blocks and root module (world_size={dist.get_world_size()})."
    )
    return model

