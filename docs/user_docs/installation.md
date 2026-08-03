# StackFormer Installation Guide

This guide covers installing StackFormer from PyPI, building from source, managing optional dependencies, and configuring developer environments.

---

## Requirements

- **Python**: `>= 3.10`
- **PyTorch**: `>= 2.0` (Supports standard CPU and CUDA execution)
- **Operating Systems**: Linux, macOS, Windows

---

## 1. Installation Options

### Option A: Install from PyPI (Recommended)

To install the latest stable package:

```bash
pip install stackformer
```

### Option B: Install from Source (Bleeding Edge)

To install the developer version directly from GitHub:

```bash
git clone https://github.com/stackformer-labs/Stackformer.git
cd Stackformer
pip install -e .
```

---

## 2. Optional Extras & Dependencies

StackFormer core maintains minimal required dependencies (`torch`, `numpy`, `tqdm`, `safetensors`). Third-party logging tools are optional:

### TensorBoard & Weights & Biases Logging

If you plan to use `TensorBoardLogger` or `WandBLogger`:

```bash
pip install tensorboard wandb
```

### Developer Dependencies (Testing & Building)

To run the test suite or build wheel distributions:

```bash
pip install "stackformer[dev]"
```

---

## 3. Verification

Verify that StackFormer installed correctly by running:

```python
import stackformer

print(f"StackFormer version: {stackformer.__version__ if hasattr(stackformer, '__version__') else '0.1.9'}")
```

Or test model instantiation:

```python
import torch
from stackformer.models import GPT2

model = GPT2(vocab_size=1000, num_layers=2, embed_dim=128, num_heads=4, seq_len=64)
x = torch.randint(0, 1000, (1, 10))
logits = model(x)
print("Forward pass successful, logits shape:", logits.shape)
```
