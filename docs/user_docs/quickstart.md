# StackFormer User Quickstart Guide

This guide walks through creating custom Transformer architectures using `BlockConfig`, initializing pre-built architectures from the model zoo, training models with the `Trainer` engine, and performing autoregressive text generation.

---

## 1. Custom Architecture Composition with `BlockConfig`

StackFormer's core design revolves around `BlockConfig`, a unified configuration object that specifies the layer primitives for Transformer backbones.

```python
import torch
from stackformer.modules import BlockConfig, TransformerEncoder

# Define block configuration
cfg = BlockConfig(
    embed_dim=256,
    num_heads=4,
    hidden_dim=1024,
    attention="gqa_rope",  # Options: "mha", "gqa", "mqa", "mha_rope", "gqa_rope", "mqa_rope", "cross_mha"
    num_kv_heads=2,
    ffn="swiglu",           # Options: "gelu", "swiglu", "geglu", "leaky_relu", "relu", "sigmoid", "silu"
    norm="rmsnorm",         # Options: "layernorm", "rmsnorm"
    pre_norm=True,
    dropout=0.1,
)

# Instantiate a 4-layer Transformer encoder stack
encoder = TransformerEncoder(cfg, num_layers=4)

# Execute forward pass with causal mask
inputs = torch.randn(2, 32, 256)
outputs = encoder(inputs, mask=True)
print("Encoder output shape:", outputs.shape)  # (2, 32, 256)
```

---

## 2. Using Pre-Built Architectures

StackFormer provides standard models pre-configured with canonical hyperparameters and block builders.

### LLaMA-2 (Grouped-Query Attention & Stateful KV Cache)

```python
import torch
from stackformer.models import Llama2

model = Llama2(
    vocab_size=32000,
    num_layers=4,
    embed_dim=512,
    num_query_heads=8,
    num_kv_heads=2,
    batch_size=1,
    kv_seq_len=256,
)

tokens = torch.randint(0, 32000, (1, 16))
logits = model(tokens, start_pos=0)
print("LLaMA-2 logits shape:", logits.shape)  # (1, 16, 32000)
```

---

## 3. Training with `Trainer`

The `Trainer` class encapsulates model optimization, evaluation loops, device placement, mixed precision, and SafeTensors checkpointing.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from stackformer.engine import Trainer
from stackformer.models import GPT2

# 1. Prepare synthetic dataset
inputs = torch.randint(0, 10000, (128, 16))
targets = torch.randint(0, 10000, (128, 16))
dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=16)

# 2. Instantiate model
model = GPT2(vocab_size=10000, num_layers=2, embed_dim=128, num_heads=4, seq_len=64)

# 3. Configure Trainer engine
trainer = Trainer(
    model=model,
    train_dataloader=dataloader,
    val_dataloader=dataloader,
    device="cpu",             # "cuda" if available
    use_amp=False,            # Set True for CUDA FP16/BF16 AMP
    max_epochs=2,
    max_train_steps=10,
    lr=5e-4,
    checkpoint_dir="checkpoints/gpt2_run",
)

# 4. Execute training fit loop
trainer.fit()
```

---

## 4. Autoregressive Text Generation

Generate sequence continuations using temperature, top-k, or top-p (nucleus) sampling via `text_generate()`:

```python
import torch
from stackformer import text_generate
from stackformer.models import Llama2

# Load model
model = Llama2(
    vocab_size=32000,
    num_layers=2,
    embed_dim=256,
    num_query_heads=4,
    num_kv_heads=2,
    batch_size=1,
    kv_seq_len=128,
)

# Prompt tensor (B, T)
prompt = torch.randint(0, 32000, (1, 5))

# Generate 15 new tokens
generated = text_generate(
    model=model,
    prompt_ids=prompt,
    max_new_tokens=15,
    temperature=0.7,
    top_p=0.9,
)

print("Generated token sequence shape:", generated.shape)  # (1, 20)
```
