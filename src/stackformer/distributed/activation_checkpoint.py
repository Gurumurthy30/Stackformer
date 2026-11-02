# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported and generalized from torchtitan/distributed/activation_checkpoint.py
# for Stackformer models and arbitrary nn.Module architectures.

from __future__ import annotations

import logging
from typing import Any, Callable
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

logger = logging.getLogger(__name__)


def _get_default_save_ops() -> set:
    """Returns the set of ops whose activations should be saved (compute + comm)."""
    compute_ops = [
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_attention_math.default,
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        torch.ops.aten.max.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.default,
    ]

    # Include higher-order ops if available
    if hasattr(torch, "_higher_order_ops") and hasattr(torch._higher_order_ops, "flex_attention"):
        compute_ops.append(torch._higher_order_ops.flex_attention)

    # Inductor compiled code if available
    if hasattr(torch, "_higher_order_ops") and hasattr(torch._higher_order_ops, "inductor_compiled_code"):
        compute_ops.append(torch._higher_order_ops.inductor_compiled_code)

    save_ops = set()
    for op in compute_ops:
        save_ops.add(op)

    # Communication ops to preserve communication-avoidance
    comm_ops = [
        getattr(torch.ops._c10d_functional, "reduce_scatter_tensor", None),
        getattr(torch.ops._c10d_functional, "all_to_all_single", None),
    ]
    for op in comm_ops:
        if op is not None and hasattr(op, "default"):
            save_ops.add(op.default)

    return save_ops


def _replace_module_in_parent(root: nn.Module, target: nn.Module, replacement: nn.Module) -> bool:
    """Find target child module in root module hierarchy and replace it with replacement."""
    for parent in root.modules():
        for name, child in parent.named_children():
            if child is target:
                if isinstance(parent, (nn.ModuleList, nn.Sequential)):
                    parent[int(name)] = replacement
                else:
                    setattr(parent, name, replacement)
                return True
    return False


def apply_activation_checkpointing(
    model: nn.Module,
    mode: str = "full",
    preserve_rng_state: bool = True,
    force_recompute_fqns: list[str] | None = None,
) -> nn.Module:
    """Apply activation checkpointing to repeating layer blocks of the model.

    Applied after Tensor Parallelism (TP) and before torch.compile and FSDP2 in the
    distributed initialization pipeline.

    Args:
        model: PyTorch nn.Module architecture.
        mode: Checkpointing mode: "none", "selective", or "full".
        preserve_rng_state: Whether to stash and restore RNG states per checkpointed block.
        force_recompute_fqns: Optional list of submodule FQN strings to force recompute.

    Returns:
        nn.Module: Model with specified repeating blocks wrapped with activation checkpointing.
    """
    mode = mode.lower().strip()
    if mode == "none":
        return model

    if mode not in {"selective", "full"}:
        raise ValueError(
            f"Unknown activation checkpointing mode '{mode}'. Expected 'none', 'selective', or 'full'."
        )

    # 1. Retrieve blocks via Stackformer protocol or auto-detection
    if hasattr(model, "get_sharding_blocks") and callable(model.get_sharding_blocks):
        blocks = model.get_sharding_blocks()
    else:
        # Fallback to scanning module lists
        blocks = []
        for mod in model.modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 1:
                types = {type(c) for c in mod}
                if len(types) == 1:
                    blocks = list(mod)
                    break
        if not blocks:
            raise ValueError(
                f"Could not identify repeating layer blocks to apply activation checkpointing for {type(model).__name__}."
            )

    save_ops = _get_default_save_ops()
    mm_ops = (
        torch.ops.aten.mm.default,
        torch.ops.aten.linear.default,
    )

    for idx, block in enumerate(blocks):
        if mode == "full":
            wrapped = checkpoint_wrapper(
                block,
                preserve_rng_state=preserve_rng_state,
            )
        elif mode == "selective":
            meta = {"forward_mm_count": 0, "recompute_mm_count": 0}

            def selective_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
                mode_str = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode_str}_mm_count"

                if func in mm_ops:
                    meta[mm_count_key] += 1
                    # Save every other matrix multiplication
                    if meta[mm_count_key] % 2 == 0:
                        return CheckpointPolicy.PREFER_RECOMPUTE
                    return CheckpointPolicy.MUST_SAVE

                if func in save_ops:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            wrapped = checkpoint_wrapper(
                block,
                context_fn=lambda: create_selective_checkpoint_contexts(selective_policy),
                preserve_rng_state=preserve_rng_state,
            )
        else:
            wrapped = block

        replaced = _replace_module_in_parent(model, block, wrapped)
        if not replaced:
            logger.warning(f"Could not replace block {idx} in parent module hierarchy.")

    logger.info(f"Applied {mode} activation checkpointing across {len(blocks)} blocks.")
    return model

