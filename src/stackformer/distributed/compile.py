# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Ported and generalized from torchtitan/distributed/compile.py
# for Stackformer models and arbitrary nn.Module architectures.

from __future__ import annotations

import logging
from typing import Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def is_compile_available() -> tuple[bool, str]:
    """Check if torch.compile is available and functional in the current environment.

    Returns:
        tuple[bool, str]: (is_available, error_reason_if_not_available)
    """
    try:
        fn = torch.compile(torch.nn.Identity())
        fn(torch.zeros(1))
        return True, ""
    except Exception as e:
        return False, str(e)


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


def apply_compile(
    model: nn.Module,
    compile_mode: str | None = None,
    backend: str = "inductor",
    per_block: bool = True,
    tp_enabled: bool = False,
    tp_mesh: Any | None = None,
) -> nn.Module:
    """Apply torch.compile to repeating blocks or the full model.

    Applied after Tensor Parallelism (TP) and Activation Checkpointing (AC),
    and before FSDP2 in the distributed execution pipeline.

    Args:
        model: PyTorch nn.Module architecture.
        compile_mode: Optional compilation optimization mode ("default", "reduce-overhead", "max-autotune").
        backend: PyTorch compiler backend (default: "inductor").
        per_block: Whether to compile individual repeating blocks rather than the entire module.
        tp_enabled: Hook for async Tensor Parallelism support.
        tp_mesh: DeviceMesh instance for TP communication if enabled.

    Returns:
        nn.Module: Compiled PyTorch module.
    """
    available, reason = is_compile_available()
    if not available:
        raise RuntimeError(
            f"torch.compile is not available in the current environment: {reason}"
        )

    # Configure Dynamo for robustness under dynamic shapes and activation checkpointing
    try:
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    except Exception:
        pass

    # Async TP hook for future multi-GPU sharding (Step 7 / Step 9)
    if tp_enabled and tp_mesh is not None:
        try:
            torch._inductor.config._micro_pipeline_tp = True
            logger.info("Async Tensor Parallelism compilation enabled.")
        except Exception:
            pass

    if per_block and hasattr(model, "get_sharding_blocks") and callable(model.get_sharding_blocks):
        blocks = model.get_sharding_blocks()
        for idx, block in enumerate(blocks):
            compiled_block = torch.compile(block, mode=compile_mode, backend=backend)
            _replace_module_in_parent(model, block, compiled_block)
        logger.info(f"Compiled {len(blocks)} blocks individually with mode={compile_mode}, backend={backend}.")
        return model

    compiled_model = torch.compile(model, mode=compile_mode, backend=backend)
    logger.info(f"Compiled full model with mode={compile_mode}, backend={backend}.")
    return compiled_model

