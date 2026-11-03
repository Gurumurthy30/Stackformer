"""Model export and serialization formats for deployment.

Planned exporters:
- ONNX (Open Neural Network Exchange)
- TorchScript / TorchDynamo AOT Export
- Safetensors consolidation
"""

from __future__ import annotations

from pathlib import Path
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    opset_version: int = 17,
) -> None:
    """Export a Stackformer model to ONNX format."""
    raise NotImplementedError(
        "ONNX export is planned for Phase 3. "
        "Use save_model_weights(model, 'model.safetensors') for current weight exports."
    )


def export_to_torchscript(
    model: nn.Module,
    output_path: str | Path,
) -> None:
    """Export a Stackformer model to TorchScript."""
    raise NotImplementedError("TorchScript export is planned for Phase 3.")


__all__ = [
    "export_to_onnx",
    "export_to_torchscript",
]

