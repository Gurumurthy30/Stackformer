# WSL2 / Linux Verification Checklist

This document tracks verification tasks, benchmarks, and tests that require a Linux or WSL2 environment with CUDA and NCCL support, which cannot be natively run in a native Windows CPU-only environment.

---

## 1. Prerequisites for WSL2 / Linux
- **OS / Runtime**: Ubuntu 22.04 LTS on WSL2 or native Linux.
- **CUDA**: NVIDIA Driver (WSL compatible) + CUDA Toolkit $\ge 12.1$.
- **PyTorch**: PyTorch $\ge 2.6.0$ built with CUDA support (`torch.cuda.is_available() == True`).
- **Distributed Backend**: NCCL (`torch.distributed.is_nccl_available() == True`).
- **Compilation Toolchain**: `gcc` / `g++` $\ge 11$, `triton` compatible with PyTorch Inductor.

---

## 2. Tests Requiring Linux / WSL2 + CUDA

| Area | Test / Task | Reason Linux / CUDA Required | Status on Windows CPU |
|---|---|---|---|
| **GPU Smoke Tests** | `tests/sanity/test_gpu_smoke.py` | Validates model forward pass and tensor allocation on CUDA devices. | Skipped (`CUDA unavailable`) |
| **Vision GPU Tests** | `tests/vision/test_vision.py:test_vision_transformer_cuda` | Tests ViT / SegFormer on GPU. | Skipped (`CUDA unavailable`) |
| **`torch.compile`** | `tests/trainer/test_step5_ac_compile_precision.py::test_compile_trainer_integration_smoke` | Requires Triton compiler and MSVC/C++ toolchain supported under Inductor. | Skipped (`triton/compiler unavailable`) |
| **Selective-AC Memory** | Peak activation memory benchmarking with `torch.cuda.max_memory_allocated()` | Verifies that `activation_checkpointing="selective"` saves memory over `"none"` without full recompute overhead. | Verified analytically; requires CUDA for byte-level memory tracking |
| **NCCL Distributed Runs** | Multi-GPU DDP, FSDP2, and TP scaling tests | Windows only supports `gloo` backend for distributed operations; `nccl` backend is Linux-only. | Tested on CPU with `gloo` |
| **Async TP Overlap** | Tensor Parallelism + `torch.compile` communication-computation overlap | Requires NCCL asynchronous collective streams and Inductor compile. | Mocked / CPU emulation only |
| **FP16 Mixed Precision** | `mixed_precision="fp16"` training loop | PyTorch `torch.amp.autocast('cpu', dtype=torch.float16)` is unsupported/unstable on CPU without hardware FP16; requires CUDA device. | Disallowed on CPU; requires CUDA |

---

## 3. How to Run Verification on WSL2

Once in WSL2 with CUDA available:

```bash
# 1. Run all skipped GPU tests
pytest tests/sanity/test_gpu_smoke.py tests/vision/test_vision.py -v

# 2. Run compilation tests
pytest tests/trainer/test_step5_ac_compile_precision.py -k "compile" -v

# 3. Benchmark Selective-AC peak memory
python -c "
import torch
from stackformer.models import GPT2
from stackformer.training import Trainer, TrainConfig
# Track memory:
print('CUDA available:', torch.cuda.is_available())
"

# 4. Multi-GPU NCCL test
torchrun --nproc_per_node=2 -m pytest tests/distributed/ -v
```
