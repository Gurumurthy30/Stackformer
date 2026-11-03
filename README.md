<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.png">
    <img src="assets/logo-light.png" alt="StackFormer logo" width="560">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/stackformer/"><img src="https://img.shields.io/pypi/v/stackformer.svg" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/stackformer/"><img src="https://img.shields.io/pypi/pyversions/stackformer.svg" alt="Python versions" /></a>
  <a href="https://github.com/stackformer-labs/Stackformer/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" /></a>
  <a href="https://github.com/stackformer-labs/Stackformer/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/stackformer-labs/Stackformer/core-tests.yml?branch=main&label=CI" alt="CI status" />
  </a>
</p>

# Stackformer

Modular PyTorch transformer blocks, clean architecture zoo, and lightweight distributed training in a single package.

---

## Why Stackformer?

Building transformer architectures typically forces researchers and engineers between two extremes: copying monolithic model scripts with tangled internal dependencies, or adopting heavy multi-layered frameworks designed around pretrained weight conversion.

Stackformer provides a third approach: composable PyTorch building blocks paired with an Accelerate-style distributed `Trainer`. Models share a unified `BlockConfig` and `BaseModel` contract, allowing users to swap attention mechanisms, feed-forward networks, normalization layers, and positional encodings without altering model math or fighting framework boilerplate.

---

## Key Features

- **Composable Layer Primitives:** Configurable multi-head, grouped-query, and cross-attention modules paired with SwiGLU, GeGLU, and standard activations.
- **Categorized Architecture Zoo:** Native PyTorch implementations of language models (GPT-1/2, LLaMA-1/2, Gemma, BERT, RoBERTa, seq2seq Transformer) and vision models (ViT, SegFormer).
- **Accelerate-Style Trainer:** Single `Trainer` interface supporting Distributed Data Parallel (DDP), Fully Sharded Data Parallel (FSDP2), mixed precision, activation checkpointing, and block-level compilation.
- **Dual Checkpointing Engine:** Clean separation between pure weights (`.safetensors` in core dependencies) and training runtime state (`state.pt` / PyTorch Distributed Checkpoint).
- **Interactive Multi-Process Launcher:** Notebook-compatible distributed launcher (`launch()`) with automatic platform configuration and worker error propagation.
- **Zero-Warning Quality Foundation:** Strictly typed codebase, zero-error Pyright compliance, Ruff linting/formatting, and Bandit security verification.

---

## Installation

Stackformer requires Python `>= 3.10` and PyTorch `>= 2.5,<3.0`.

### Core Package

```bash
pip install stackformer
```

### Optional Extras

```bash
# Training utilities (tqdm progress bars, rich console display)
pip install "stackformer[train]"

# Experiment tracking loggers
pip install "stackformer[tensorboard]"
pip install "stackformer[wandb]"

# Low-precision Float8 GEMMs via torchao
pip install "stackformer[float8]"

# Full development & test suite
pip install "stackformer[dev]"
```

### Install from Source

```bash
git clone https://github.com/stackformer-labs/Stackformer.git
cd Stackformer
pip install -e .
```

---

## Quick Start

### 1. Build a Model Architecture

```python
import torch
from stackformer.models.language import GPT2

# Instantiate a compact GPT-2 model
model = GPT2(
    vocab_size=1000,
    num_layers=4,
    embed_dim=128,
    num_heads=4,
    seq_len=64,
)

x = torch.randint(0, 1000, (2, 32))
logits = model(x)
print(f"Logits shape: {logits.shape}")  # torch.Size([2, 32, 1000])
```

### 2. Train with `Trainer` and `TrainConfig`

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from stackformer.models.language import GPT2
from stackformer.training import Trainer, TrainConfig

# Prepare model and synthetic dataset
model = GPT2(vocab_size=500, num_layers=2, embed_dim=64, num_heads=4, seq_len=32)
dataset = TensorDataset(torch.randint(0, 500, (64, 16)))
loader = DataLoader(dataset, batch_size=8)

# Configure training parameters
config = TrainConfig(
    max_steps=10,
    lr=1e-3,
    mixed_precision="no",  # "no", "bf16", or "fp16"
    checkpoint_dir="./runs/quickstart",
)

trainer = Trainer(model=model, train_loader=loader, config=config)
trainer.train()
```

### 3. Save and Reload Checkpoints

```python
import stackformer
from stackformer.models.language import GPT2

# Save weights via Trainer
trainer.save("./runs/quickstart/model.safetensors")

# Load weights into a fresh model instance
fresh_model = GPT2(vocab_size=500, num_layers=2, embed_dim=64, num_heads=4, seq_len=32)
stackformer.load_weights(fresh_model, "./runs/quickstart/model.safetensors")
```

### 4. Distributed Multi-Process Training with `launch()`

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from stackformer.models.language import GPT2
from stackformer.training import Trainer, TrainConfig, launch

def run_worker():
    model = GPT2(vocab_size=500, num_layers=2, embed_dim=64, num_heads=4, seq_len=32)
    loader = DataLoader(TensorDataset(torch.randint(0, 500, (32, 16))), batch_size=8)
    config = TrainConfig(max_steps=5, lr=1e-3, parallel="ddp")
    trainer = Trainer(model=model, train_loader=loader, config=config)
    trainer.train()

if __name__ == "__main__":
    # Launch 2-process DDP execution (works in scripts and Jupyter notebooks)
    launch(run_worker, num_processes=2, backend="gloo")
```

### 5. Autoregressive Text Generation

```python
import torch
from stackformer.models.language import GPT2
from stackformer import text_generate

model = GPT2(vocab_size=500, num_layers=2, embed_dim=64, num_heads=4, seq_len=32)
prompt = torch.tensor([[10, 25, 42]])

tokens = text_generate(
    model,
    prompt,
    max_new_tokens=10,
    temperature=0.8,
    top_k=50,
)
print("Generated token IDs:", tokens.tolist())
```

---

## Model Zoo

Models are organized by category and can be imported from domain packages (`stackformer.models.language`, `stackformer.models.vision`) or top-level `stackformer.models`.

| Category | Model Class | Family / Paper | Description |
| :--- | :--- | :--- | :--- |
| **Language** | `GPT1`, `GPT2` | OpenAI GPT | Causal language models with learned positional embeddings |
| | `Llama1`, `Llama2` | Meta LLaMA | Autoregressive models with RMSNorm, SwiGLU, GQA, and RoPE |
| | `Gemma1_2B`, `Gemma1_7B` | Google Gemma | Causal models with GeGLU, unit-offset RMSNorm, and RoPE |
| | `BERT` | Bidirectional Encoder | Bidirectional transformer encoder with absolute position embeddings |
| | `RoBERTa` | Robustly Optimized BERT | Bidirectional encoder with dynamic masking configuration |
| | `Transformer` | Vaswani Seq2Seq | Original encoder-decoder architecture with cross-attention |
| **Vision** | `ViT` | Vision Transformer | Patch projection encoder with classification head |
| | `SegFormer` | Efficient SegFormer | Hierarchical vision transformer with sequence reduction attention |
| **Planned** | Multimodal | Vision-Language | Cross-modal projection layers and vision-text decoders (CLIP, LLaVA) |
| | Audio | Speech Transformers | Audio spectrogram patchifiers and speech decoders (Whisper) |

### Model Registry

```python
from stackformer.models import list_models, get_model

# Query available architectures grouped by category
print(list_models())
# {'language': ['BERT', 'GPT1', 'GPT2', ...], 'vision': ['SegFormer', 'ViT']}

# Factory instantiation by name
model = get_model("gpt2", vocab_size=1000, num_layers=4, embed_dim=128, num_heads=4, seq_len=64)
```

---

## Training Capabilities

| Feature | Trainer Setting | Validation Status | Description |
| :--- | :--- | :--- | :--- |
| **Distributed Data Parallel** | `parallel="ddp"` | Verified | PyTorch DDP on Gloo (CPU) and NCCL (CUDA) |
| **FSDP2 Sharding** | `parallel="fsdp2"` | Experimental | Native `torch.distributed.fsdp.fully_shard` (CUDA numerical parity pending) |
| **Activation Checkpointing** | `activation_checkpointing="full"` | Verified | Checkpoints repeating block forward passes to minimize activation memory |
| **Per-Block Compilation** | `compile=True` | Supported | Per-block `torch.compile` integration (CUDA validation pending) |
| **Mixed Precision** | `mixed_precision="bf16"` | Verified | Native PyTorch AMP (`torch.amp.autocast`) with GradScaler |
| **Distributed Checkpoint** | `use_dcp=True` | Verified | PyTorch DCP integration for parallel sharded state persistence |
| **Mid-Epoch Resumption** | `resume_from="path"` | Verified | Bit-identical resumption restoring optimizer, scheduler, and RNG state |
| **Best Model Tracking** | `save_best=True` | Verified | Automated metric monitoring (e.g. `val_loss`) with checkpoint pruning |
| **Display & Observability** | `display="rich"` | Verified | Pretty Rich formatting with rank-0 gating and clean plain fallback |
| **Multi-Backend Loggers** | `loggers=[...]` | Verified | Integrated Console, CSV, JSONL, TensorBoard, and Weights & Biases |
| **Loss Registry** | `loss="cross_entropy"` | Verified | Extensible loss functions via `register_loss()` with automatic mode routing |
| **Lifecycle Callbacks** | `callbacks=[...]` | Verified | Event hooks for step/epoch boundaries, early stopping, and LR logging |

---

## Checkpoint Security

Stackformer checkpoint directories separate weights from execution state:
- **`model.safetensors`**: Pure model weights serialized in Hugging Face Safetensors format. Free from arbitrary code execution and memory-mapped for fast loading.
- **`state.pt`**: Contains optimizer tensors, learning rate scheduler state, step counters, and Python / NumPy RNG states required for exact mid-epoch resumption.

> [!WARNING]
> Because `state.pt` restores NumPy RNG state and optimizer state dictionaries, it is loaded with `torch.load(..., weights_only=False)`. **Only resume training from checkpoints generated by trusted runs.** When loading weights solely for evaluation, fine-tuning, or inference, load `model.safetensors` via `stackformer.load_weights()`.

---

## Project Structure

```
stackformer/
├── src/
│   └── stackformer/
│       ├── models/             # Domain model architectures
│       │   ├── language/       # GPT, LLaMA, Gemma, BERT, RoBERTa, Transformer
│       │   ├── vision/         # ViT, SegFormer
│       │   ├── multimodal/     # Vision-language architectures (planned)
│       │   ├── audio/          # Speech and audio transformers (planned)
│       │   └── base.py         # BaseModel contract and lifecycle hooks
│       ├── modules/            # Composable attention, FFN, norm, layers, and masks
│       ├── training/           # Accelerate-style Trainer, callbacks, loggers, losses
│       ├── distributed/        # DDP, FSDP2 sharding, DCP checkpointing, launcher
│       ├── ops/                # Optimized operations (scatter_add)
│       ├── quantization/       # Float8 linear modules & low-precision hooks
│       ├── cache/              # Static & paged KV cache implementations (experimental)
│       ├── initialization/     # Parameter initialization strategies
│       ├── utils/              # Weight checkpointing and attention helpers
│       ├── data/               # Streaming dataset and collation (planned)
│       ├── peft/               # Parameter-efficient fine-tuning (planned)
│       ├── export/             # ONNX and deployment serialization (planned)
│       ├── eval/               # Evaluation harnesses and benchmark tasks (planned)
│       └── serving/            # Continuous batching inference server (planned)
├── examples/                   # 5 runnable CPU notebooks (GPT-2, BERT, ViT, SegFormer, Seq2Seq)
├── benchmarks/                 # Throughput and memory scaling benchmarks (planned)
├── docs/                       # Documentation site, roadmap, and guides
├── tests/                      # 270+ unit, integration, and distributed tests
└── pyproject.toml              # Single source of truth configuration
```

---

## Documentation

- **Getting Started:**
  - [Installation Guide](docs/getting-started/installation.md)
  - [Quickstart Guide](docs/getting-started/quickstart.md)
  - [First Transformer Tutorial](docs/getting-started/first-transformer.md)
- **Guides & Reference:**
  - [Training & Checkpointing Guide](docs/training.md)
  - [Architecture Overview](docs/fundamentals/architecture-overview.md)
  - [Strategic Roadmap](docs/ROADMAP.md)
  - [Planned Subsystems Index](docs/PLANNED.md)
- **Developer Guide:**
  - [Architecture & Design Principles](docs/developer-guide/architecture.md)
  - [Contributing Guidelines](docs/developer-guide/contributing.md)

---

## Contributing

Contributions are welcome. Please check the [Contributing Guide](docs/developer-guide/contributing.md) and [Strategic Roadmap](docs/ROADMAP.md) before submitting pull requests.

---

## Citation

If you use Stackformer in your research or project, please cite:

```bibtex
@software{stackformer2026,
  author = {Stackformer Labs},
  title = {Stackformer: Modular PyTorch Transformer Architecture Zoo and Distributed Trainer},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/stackformer-labs/Stackformer}}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

Stackformer contains code derived from [TorchTitan](https://github.com/pytorch/torchtitan) (Meta Platforms, Inc.), licensed under the BSD 3-Clause License. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSE-torchtitan`](LICENSE-torchtitan) for details. Stackformer is an independent project and is not affiliated with or endorsed by Meta Platforms, Inc.
