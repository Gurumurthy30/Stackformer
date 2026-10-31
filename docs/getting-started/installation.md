# Installation Guide

This guide covers installing StackFormer from PyPI, building from source for development, configuring optional dependencies, and verifying your installation.

---

## 1. System Requirements

Before installing StackFormer, ensure your environment meets the following specifications:

- **Python**: `>= 3.10` (Tested on 3.10, 3.11, and 3.12).
- **PyTorch**: `>= 2.0` (StackFormer relies on native `torch.nn` and `F.scaled_dot_product_attention`).
- **Operating Systems**: Linux (Ubuntu 20.04+), macOS (Apple Silicon / Intel), Windows 10/11.
- **Hardware**: Compatible with CPU, Apple Silicon (MPS), and NVIDIA CUDA GPUs.

---

## 2. Installing PyTorch

StackFormer requires PyTorch. If you do not have PyTorch installed yet, install the appropriate variant for your hardware:

=== "CUDA 12.4 (NVIDIA GPU)"
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    ```

=== "CUDA 12.1 (NVIDIA GPU)"
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

=== "CPU-Only (Linux / Windows)"
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

=== "macOS (Apple Silicon MPS)"
    ```bash
    pip install torch
    ```

---

## 3. Installing StackFormer

### Option A: Install from PyPI (Recommended)

To install the latest stable version of StackFormer:

```bash
pip install --upgrade pip
pip install stackformer
```

### Option B: Install from Source (Development Mode)

If you are contributing to StackFormer, modifying the source code, or following bleeding-edge features:

```bash
# 1. Clone the repository
git clone https://github.com/stackformer-labs/Stackformer.git
cd Stackformer

# 2. Install in editable development mode with test toolchain
pip install -e ".[dev]"
```

The `.[dev]` extra installs:
- `pytest>=8.0`: Test runner
- `pytest-cov>=5.0`: Coverage reporting
- `build>=1.2`: Package distribution builder

---

## 4. Optional Extras

StackFormer maintains a lean core dependency footprint (`torch`, `numpy`, `tqdm`, `safetensors`). Experiment loggers are optional:

### TensorBoard & Weights & Biases

To enable experiment tracking with `TensorBoardLogger` or `WandBLogger`:

```bash
pip install tensorboard wandb
```

### Documentation Toolchain

To build this documentation website locally:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions
```

---

## 5. Verification

To verify that StackFormer and its dependencies are installed and working properly, run:

```bash
python -c "import stackformer; print('StackFormer version:', getattr(stackformer, '__version__', 'installed')); import torch; print('PyTorch version:', torch.__version__)"
```

### Smoke Test: Instantiate a Model

Run a quick test to instantiate a Transformer and verify a forward pass:

```python
import torch
from stackformer.models import GPT2

# 1. Instantiate a small GPT-2 model
model = GPT2(
    vocab_size=1000,
    num_layers=2,
    embed_dim=128,
    num_heads=4,
    seq_len=64,
)

# 2. Run a forward pass
x = torch.randint(0, 1000, (2, 16))
logits = model(x)

print(f"Success! Logits output tensor shape: {logits.shape}")  # torch.Size([2, 16, 1000])
```

---

## 6. Common Installation Pitfalls

### Missing CUDA Headers on Windows

If you see `UserWarning: Failed to find CUDA. Install CUDA Toolkit...`, this is emitted by Triton's Windows utility. StackFormer's core models and training engine will continue executing normally on CPU or PyTorch native CUDA. To resolve, install the full NVIDIA CUDA Toolkit and set `CUDA_HOME` in your environment.

### Pip Build Isolation Failures in Air-Gapped Environments

If your environment cannot access the internet during editable installs:

```bash
pip install --no-build-isolation -e .
```
