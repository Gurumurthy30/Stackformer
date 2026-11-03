# WSL2 / Linux Verification Checklist

This document tracks verification tasks, benchmarks, release blockers, and tests that require a Linux or WSL2 environment with CUDA and NCCL support, which cannot be natively run in a native Windows CPU-only environment.

---

## 1. Prerequisites for WSL2 / Linux
- **OS / Runtime**: Ubuntu 22.04 LTS on WSL2 or native Linux.
- **CUDA**: NVIDIA Driver (WSL compatible) + CUDA Toolkit $\ge 12.1$.
- **PyTorch**: PyTorch $\ge 2.6.0$ built with CUDA support (`torch.cuda.is_available() == True`).
- **Distributed Backend**: NCCL (`torch.distributed.is_nccl_available() == True`).
- **Compilation Toolchain**: `gcc` / `g++` $\ge 11$, `triton` compatible with PyTorch Inductor.

---

## 2. Release Blockers Requiring Linux / WSL2 + CUDA

The following items are designated release blockers to be verified once in a CUDA/WSL2 environment:

1. **FSDP2 Numerical Parity vs Single-Process**:
   - Verify that training under `parallel="fsdp2"` produces identical / tolerance-equivalent losses and gradient updates to single-process training.
   - *Reason:* PyTorch 2.6's `torch.distributed.fsdp.fully_shard` unconditionally requires device streams with event recording (`Stream.record_event`), which CPU streams do not support.
2. **FSDP2 Distributed Checkpoint (DCP) Save & Resume**:
   - Verify that an FSDP2-sharded model can save via DCP and resume with exact numerical continuity and identical next-step loss.
3. **DCP to SafeTensors on a Sharded Model**:
   - Consolidate an FSDP2 DCP checkpoint directory into a single unsharded `model.safetensors` file and verify that `BaseModel.load_weights()` produces identical outputs.
4. **Multi-GPU NCCL Distributed Runs**:
   - Execute multi-GPU scaling tests under `backend="nccl"`, validating DDP, FSDP2, and Tensor Parallelism across physical GPU devices.

---

## 3. Other Tests Requiring Linux / WSL2 + CUDA

| Area | Test / Task | Reason Linux / CUDA Required | Status on Windows CPU |
|---|---|---|---|
| **GPU Smoke Tests** | `tests/sanity/test_gpu_smoke.py` | Validates model forward pass and tensor allocation on CUDA devices. | Skipped (`CUDA unavailable`) |
| **Vision GPU Tests** | `tests/vision/test_vision.py:test_vision_transformer_cuda` | Tests ViT / SegFormer on GPU. | Skipped (`CUDA unavailable`) |
| **`torch.compile`** | `tests/trainer/test_step5_ac_compile_precision.py::test_compile_trainer_integration_smoke` | Requires Triton compiler and MSVC/C++ toolchain supported under Inductor. | Skipped (`triton/compiler unavailable`) |
| **Selective-AC Memory** | Peak activation memory benchmarking with `torch.cuda.max_memory_allocated()` | Verifies that `activation_checkpointing="selective"` saves memory over `"none"` without full recompute overhead. | Verified analytically; requires CUDA for byte-level memory tracking |
| **Async TP Overlap** | Tensor Parallelism + `torch.compile` communication-computation overlap | Requires NCCL asynchronous collective streams and Inductor compile. | Emulated on CPU |
| **FP16 Mixed Precision** | `mixed_precision="fp16"` training loop | PyTorch `torch.amp.autocast('cpu', dtype=torch.float16)` is unsupported/unstable on CPU without hardware FP16; requires CUDA device. | Disallowed on CPU; requires CUDA |

---

## 4. How to Run Verification on WSL2

Once in WSL2 with CUDA available:

```bash
# 1. Run all skipped GPU tests
pytest tests/sanity/test_gpu_smoke.py tests/vision/test_vision.py -v

# 2. Run compilation tests
pytest tests/trainer/test_step5_ac_compile_precision.py -k "compile" -v

# 3. Multi-GPU NCCL & FSDP2 tests
torchrun --nproc_per_node=2 -m pytest tests/distributed/ -v
```
