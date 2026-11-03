# Model Export Subsystem

## Overview
The `stackformer.export` module provides standardized tools for compiling and serializing trained Stackformer models into deployment runtimes:
- **Safetensors**: Lightweight, zero-copy, secure tensor weights (supported today via `save_model_weights` and `Trainer.save`).
- **ONNX**: Cross-platform open graph format for inference engines like ONNX Runtime, TensorRT, and OpenVINO.
- **TorchDynamo AOT**: Pre-compiled PyTorch export bundles suitable for C++ libtorch environments.
