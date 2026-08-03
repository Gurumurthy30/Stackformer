<p align="center">
  <img src="assets/logo.png" alt="StackFormer logo" width="560" />
</p>

<p align="center">
  <a href="https://pypi.org/project/stackformer/"><img src="https://img.shields.io/pypi/v/stackformer.svg" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/stackformer/"><img src="https://img.shields.io/pypi/pyversions/stackformer.svg" alt="Python versions" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" /></a>
  <a href="https://github.com/stackformer-labs/Stackformer/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/stackformer-labs/Stackformer/core-tests.yml?branch=main&label=CI" alt="CI status" />
  </a>
</p>

# StackFormer

**Composable PyTorch Transformer building blocks, modular architecture zoo, and lightweight training engine in a single clean library.**

---

## Why StackFormer?

Building custom Transformer architectures today usually forces a frustrating choice: either fork thousands of lines of monolithic, copy-pasted model code from single-file implementations, or navigate bloated multi-layer abstraction frameworks designed exclusively for pretrained weight conversion.

**StackFormer** provides a third way: modular PyTorch building blocks that let researchers and engineers assemble custom vision and language Transformers declaratively via `BlockConfig`. Instead of rewriting multi-head attention, rotary positional embeddings, SwiGLU feed-forward networks, or pre-norm connections for every experiment, StackFormer exposes orthogonal layer primitives that compose cleanly and execute on pure PyTorch runtime.

---

## Key Features

- **Declarative `BlockConfig` Layer Composability:** Arbitrarily combine 10 attention implementations (`mha`, `gqa`, `mqa`, `cross_mha`, and RoPE variants), 7 feed-forward activation layers (`gelu`, `swiglu`, `geglu`, `leaky_relu`, `relu`, `sigmoid`, `silu`), 2 normalization types (`layernorm`, `rmsnorm`), and 3 positional embedding strategies (`absolute`, `sinusoidal`, `rope`).
- **Comprehensive Architecture Zoo (Language & Vision):** Production-faithful PyTorch implementations of **GPT-1, GPT-2, LLaMA-1, LLaMA-2, Gemma-1 (2B & 7B), BERT, RoBERTa, Vaswani Encoder-Decoder Transformer, ViT, and SegFormer-B0**.
- **Lightweight Engine & Trainer:** Built-in `Trainer` featuring Automatic Mixed Precision (`AMPScaler` for FP16/BF16), Distributed Data Parallel (`DDP`), gradient accumulation, max gradient clipping, warmup/decay learning rate schedulers, and zero-dependency SafeTensors checkpointing (`CheckpointManager`).
- **Stateful KV-Cache Autoregressive Generation:** Standardized `prefill()` and `decode()` model contract supporting fast KV-cache accelerated text generation (`text_generate()`) with temperature, top-k, and top-p (nucleus) sampling strategies.
- **Zero-Bloat Foundation:** Typed codebase (`py.typed`), 136-test suite across unit and integration targets, with zero heavy required dependencies beyond PyTorch, NumPy, SafeTensors, and tqdm.

---

## Framework Comparison

| Dimension | StackFormer | Hugging Face Transformers | nanoGPT | x-transformers |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Focus** | Modular custom block composition & native training engine | Pretrained weight distribution & pipeline inference | Minimal teaching codebase for GPT-2 | Experimental attention module collection |
| **Architecture Scope** | Language + Vision (GPT, LLaMA, Gemma, BERT, RoBERTa, ViT, SegFormer) | Comprehensive pretrained repository | Decoder-only GPT models | Modular Transformer blocks |
| **Block Composability** | Declarative `BlockConfig` (swap attention, FFN, norm, pos-emb seamlessly) | Monolithic per-model source files | Single-file script | PyTorch module blocks |
| **Training Infrastructure** | Built-in `Trainer` (AMP, DDP, SafeTensors, gradient accum, schedulers) | Requires `Trainer` or Accelerate | Custom training loop in `train.py` | External (user writes loop) |
| **Checkpoint Format** | SafeTensors (`.safetensors`) + JSON metadata | SafeTensors / PyTorch bin | PyTorch `.pt` dictionary | External |

---

## Installation

### Prerequisites
- Python `>= 3.10`
- PyTorch `>= 2.0`

### Install via PyPI

```bash
pip install stackformer
```

### Install from Source

```bash
git clone https://github.com/stackformer-labs/Stackformer.git
cd Stackformer
pip install -e .
```

### Optional Dependencies

For logging integrations (TensorBoard / Weights & Biases):

```bash
pip install tensorboard wandb
```

---

## Quick Start

### 1. Build a Custom Architecture with `BlockConfig`

```python
import torch
from stackformer.modules import BlockConfig, TransformerEncoder

# Define a custom Transformer block configuration
config = BlockConfig(
    embed_dim=512,
    num_heads=8,
    hidden_dim=2048,
    attention="gqa_rope",  # Grouped-Query Attention with Rotary Embeddings
    num_kv_heads=2,
    ffn="swiglu",           # SwiGLU Feed-Forward Network
    norm="rmsnorm",         # RMSNorm normalization
    pre_norm=True,
    dropout=0.1,
)

# Instantiate a 6-layer Encoder backbone
encoder = TransformerEncoder(config, num_layers=6)

x = torch.randn(2, 64, 512)
output = encoder(x, mask=True)
print("Encoder output shape:", output.shape)  # torch.Size([2, 64, 512])
```

### 2. Instantiate a Model from the Architecture Zoo

```python
import torch
from stackformer.models import Llama2

# Initialize LLaMA-2 with GQA and stateful KV-cache support
model = Llama2(
    vocab_size=32000,
    num_layers=4,
    embed_dim=512,
    num_query_heads=8,
    num_kv_heads=2,
    batch_size=1,
    kv_seq_len=128,
)

input_ids = torch.randint(0, 32000, (1, 16))
logits = model(input_ids, start_pos=0)
print("Logits shape:", logits.shape)  # torch.Size([1, 16, 32000])
```

### 3. Train with the High-Level `Trainer` Engine

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from stackformer.engine import Trainer
from stackformer.models import GPT2

# Synthetic dataset
x = torch.randint(0, 50257, (64, 16))
y = torch.randint(0, 50257, (64, 16))
train_loader = DataLoader(TensorDataset(x, y), batch_size=8)

model = GPT2(vocab_size=50257, num_layers=2, embed_dim=256, num_heads=4, seq_len=64)

trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=train_loader,
    device="cpu",
    use_amp=False,
    use_ddp=False,
    max_epochs=1,
    max_train_steps=5,
    lr=3e-4,
    checkpoint_dir="checkpoints",
)

trainer.fit()
```

### 4. Autoregressive Text Generation

```python
import torch
from stackformer import text_generate
from stackformer.models import Llama2

model = Llama2(
    vocab_size=32000,
    num_layers=2,
    embed_dim=256,
    num_query_heads=4,
    num_kv_heads=2,
    batch_size=1,
    kv_seq_len=128,
)

prompt = torch.randint(0, 32000, (1, 8))
generated_ids = text_generate(
    model=model,
    prompt_ids=prompt,
    max_new_tokens=20,
    temperature=0.8,
    top_p=0.9,
)
print("Generated shape:", generated_ids.shape)  # torch.Size([1, 28])
```

---

## Project Structure

```text
Stackformer/
├── assets/                       # Branding logos and documentation images
├── ci/                           # CI runner scripts (Kaggle GPU integration)
├── docs/                         # User & developer documentation
│   ├── roadmap.md                # Detailed technical roadmap (Phases 0–5)
│   ├── user_docs/                # Installation, quickstart, API reference
│   └── developer_docs/           # Architecture deep-dive and scope/design non-goals
├── examples/                     # Runnable usage examples
│   ├── simple_trainer.py         # Trainer engine execution demo
│   └── train_ddp.py              # Multi-GPU DistributedDataParallel (DDP) demo
├── reviews/                      # Internal engineering reviews and roadmap specifications
├── stackformer/                  # Core library package
│   ├── __init__.py               # Top-level API exports
│   ├── config.py                 # ModelConfig, TrainingConfig, GenerationConfig dataclasses
│   ├── generate.py               # Autoregressive decoding engine and KV-cache dispatcher
│   ├── metrics.py                # Public metric utilities
│   ├── py.typed                  # PEP 561 inline typing indicator
│   ├── amp/                      # Automatic Mixed Precision (AMPScaler)
│   ├── cache/                    # KV-cache strategies (StaticKVCache, PagedKVCache scaffold)
│   ├── distributed/              # DistributedDataParallel (DDP) wrappers and process helpers
│   ├── engine/                   # High-level Trainer, Engine, State, and CheckpointManager
│   ├── language/                 # Abstract decoder and encoder-decoder bases
│   ├── logging/                  # Metrics tracking, CSV, TensorBoard, WandB loggers
│   ├── models/                   # GPT-1/2, LLaMA-1/2, Gemma-1, BERT, RoBERTa, Transformer
│   ├── modules/                  # Attention, FFN, Norm, Positional Embedding, BlockConfig, Layer
│   ├── optim/                    # Optimizer and scheduler factory constructors and loss functions
│   ├── training/                 # Engine loop helper routines
│   ├── utils/                    # Device helpers, seed utility, shape formatting
│   └── vision/                   # ViT and SegFormer-B0 architectures
└── tests/                        # 136 passing tests across unit, integration, and model suites
```

---

## Architecture & Model Zoo

StackFormer ships verified implementations of canonical language and vision models built directly on `stackformer.modules`:

| Model Architecture | Category | Attention Mechanism | Positional Embedding | Normalization | FFN Activation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-1** | Causal Language Model | MHA (Causal) | Absolute Learned | Post-LayerNorm | GELU |
| **GPT-2** | Causal Language Model | MHA (Causal) | Absolute Learned | Pre-LayerNorm | GELU |
| **LLaMA-1** | Causal Language Model | MHA + RoPE | RoPE | Pre-RMSNorm | SwiGLU |
| **LLaMA-2** | Causal Language Model | GQA + Stateful KV Cache | RoPE | Pre-RMSNorm | SwiGLU |
| **Gemma-1 (2B/7B)** | Causal Language Model | MQA / GQA + RoPE | RoPE | Pre-RMSNorm | GeGLU |
| **BERT** | Bidirectional Language Model | MHA (Bidirectional) | Absolute Learned + Segment | Post-LayerNorm | GELU |
| **RoBERTa** | Bidirectional Language Model | MHA (Bidirectional) | Absolute Learned Offset | Post-LayerNorm | GELU |
| **Vaswani Transformer** | Seq2Seq Encoder-Decoder | Causal MHA + Cross-MHA | Sinusoidal Fixed | Post-LayerNorm | ReLU |
| **ViT** | Vision Classification | MHA (Bidirectional) | Absolute Learned | Pre-LayerNorm | GELU |
| **SegFormer-B0** | Semantic Segmentation | Spatial Reduction Attention | Efficient Mix-FFN | Pre-LayerNorm | GELU |

---

## Roadmap

Development is structured into linear engineering phases per [`reviews/00_FUTURE_PLAN.md`](docs/roadmap.md):

- **Phase 0 — Unblock Core Generation (Current):** Standardize `prefill()` and `decode()` KV-cache contracts across decoder architectures, update `text_generate()`, and enforce cache parity testing. *(Completed for LLaMA-2).*
- **Phase 1 — Attention Engine & Compiler Integration (Planned):** Unified `AttentionEngine` kernel dispatcher (`stackformer/modules/attention_engine.py`) routing between SDPA, FlexAttention, `flash-attn` v2/v4, and custom Triton fallbacks with zero `torch.compile` graph breaks.
- **Phase 2 — Budget Planner & PyTorch Native FSDP2 Scaling (Planned):** Analytical and empirical execution planner (`plan_training()`), native PyTorch FSDP2 sharding (`torch.distributed.fsdp.fully_shard`) + `DTensor` + Distributed Checkpoint (`DCP`).
- **Phase 3 — `torchao` Quantization & Native Adapters (Planned):** Native low-precision quantization via `torchao` (INT8, INT4, FP8, NF4) and zero-bloat `LoRALinear` / `DoRALinear` modules on `BlockConfig`.
- **Phase 4 — Multimodal Vision-Language & Document AI (Planned):** Unified vision-text backbone combining SegFormer/ViT patch encoders with LLaMA decoder stacks via cross-attention and projection adapters.
- **Phase 5 — Paged KV Cache & Serving Engine (Planned):** `KVCacheManager` featuring PagedAttention virtual block tables and high-concurrency continuous-batching server (`FastAPI` + Model Context Protocol server).

*For the complete deep technical specification, see [docs/roadmap.md](docs/roadmap.md).*

---

## Documentation

- **User Documentation:**
  - [Installation Guide](docs/user_docs/installation.md)
  - [Quickstart Guide](docs/user_docs/quickstart.md)
  - [API Reference](docs/user_docs/api_reference.md)
- **Developer & Technical Documentation:**
  - [Architecture Specification](docs/developer_docs/architecture.md)
  - [Library Scope & Design Non-Goals](docs/developer_docs/scope.md)
  - [Technical Roadmap](docs/roadmap.md)
- **Code Examples:**
  - [Runnable Example Scripts](examples/)

---

## Community & Resources

- **GitHub Repository:** [https://github.com/stackformer-labs/Stackformer](https://github.com/stackformer-labs/Stackformer)
- **Issue Tracker:** [https://github.com/stackformer-labs/Stackformer/issues](https://github.com/stackformer-labs/Stackformer/issues)
- **Discussions:** [https://github.com/stackformer-labs/Stackformer/discussions](https://github.com/stackformer-labs/Stackformer/discussions)
- **Releases:** [https://github.com/stackformer-labs/Stackformer/releases](https://github.com/stackformer-labs/Stackformer/releases)

---

## Contributing

We welcome community contributions to StackFormer! Please review our [Library Scope & Non-Goals](docs/developer_docs/scope.md) before opening pull requests to ensure alignment with project design principles.

---

## Citation

If you use StackFormer in your research or project, please consider citing:

```bibtex
@software{stackformer2026,
  author = {Stackformer Labs},
  title = {StackFormer: A Modular PyTorch Framework for Transformer Architecture Composability},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/stackformer-labs/Stackformer}}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
