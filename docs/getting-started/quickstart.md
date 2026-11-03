# Quickstart Guide

This 5-minute guide walks through StackFormer's core capabilities:
1. Building custom architectures with `BlockConfig`.
2. Initializing pre-built models from the architecture zoo.
3. Training a model with standard PyTorch.
4. Generating text with the autoregressive decoding engine.

---

## 1. Custom Architecture Composition with `BlockConfig`

StackFormer replaces monolithic model code with declarative block configuration. Swapping attention backends or feed-forward activations takes a single parameter change:

```python
import torch
from stackformer.modules import BlockConfig, TransformerEncoder

# 1. Configure the layer architecture
config = BlockConfig(
    embed_dim=256,
    num_heads=4,
    hidden_dim=1024,
    attention="gqa_rope",  # Grouped-Query Attention with Rotary Embeddings
    num_kv_heads=2,        # 2 KV heads (ratio: 2 queries per KV head)
    ffn="swiglu",          # SwiGLU Feed-Forward Network
    norm="rmsnorm",        # RMSNorm normalization
    pre_norm=True,         # Pre-normalization (GPT/LLaMA style)
    dropout=0.1,
)

# 2. Instantiate a 4-layer Encoder stack
encoder = TransformerEncoder(config, num_layers=4)

# 3. Execute forward pass with causal mask
inputs = torch.randn(2, 32, 256)  # (Batch=2, Sequence=32, Embed=256)
outputs = encoder(inputs, mask=True)

print("Encoder output shape:", outputs.shape)
# Output: torch.Size([2, 32, 256])
```

---

## 2. Using Pre-Built Architectures from the Model Zoo

StackFormer provides production-faithful implementations of canonical architectures. Each model constructor accepts hyperparameter arguments directly:

### LLaMA-2 (GQA + RoPE + RMSNorm)

```python
import torch
from stackformer.models.language import Llama2

# Instantiate a compact LLaMA-2 model
model = Llama2(
    vocab_size=32000,
    num_layers=4,
    embed_dim=512,
    num_query_heads=8,
    num_kv_heads=2,
    seq_len=256,
)

# Run forward pass on token IDs
tokens = torch.randint(0, 32000, (1, 16))
logits = model(tokens)

print("LLaMA-2 logits shape:", logits.shape)
# Output: torch.Size([1, 16, 32000])
```

### GPT-2 (MHA + Post-LayerNorm + Absolute Embeddings)

```python
from stackformer.models.language import GPT2

gpt = GPT2(
    vocab_size=50257,
    num_layers=6,
    embed_dim=384,
    num_heads=6,
    seq_len=512,
)

tokens = torch.randint(0, 50257, (2, 32))
logits = gpt(tokens)
print("GPT-2 logits shape:", logits.shape)
# Output: torch.Size([2, 32, 50257])
```

---

## 3. Training with Standard PyTorch

StackFormer models are 100% standard `torch.nn.Module` instances, integrating seamlessly into standard PyTorch loops, Hugging Face Accelerate, or custom training engines:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stackformer.models.language import GPT2

# 1. Prepare synthetic dataset (tokens and labels)
inputs = torch.randint(0, 10000, (128, 16))
targets = inputs.clone()
dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=16)

# 2. Instantiate model and optimizer
model = GPT2(vocab_size=10000, num_layers=2, embed_dim=128, num_heads=4, seq_len=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss()

# 3. Execute training loop
model.train()
for epoch in range(2):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits.view(-1, 10000), batch_y.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} complete. Step Loss: {loss.item():.4f}")
```

---

## 4. Autoregressive Text Generation

Generate sequence continuations using `text_generate()`, supporting temperature scaling, top-k filtering, and nucleus (top-p) sampling:

```python
import torch
from stackformer import text_generate
from stackformer.models.language import GPT2

# 1. Instantiate model
model = GPT2(vocab_size=10000, num_layers=2, embed_dim=128, num_heads=4, seq_len=64)

# 2. Prompt token sequence (Batch=1, Length=5)
prompt = torch.randint(0, 10000, (1, 5))

# 3. Generate 15 new tokens
generated = text_generate(
    model=model,
    prompt_ids=prompt,
    max_new_tokens=15,
    temperature=0.7,
    top_p=0.9,
)

print("Generated token sequence shape:", generated.shape)
# Output: torch.Size([1, 20]) (5 prompt tokens + 15 generated tokens)
```

---

## Next Steps

- Learn how to assemble complete custom backbones: [Build Your First Transformer](first-transformer.md)
- Explore `BlockConfig` parameters in depth: [Block Composition Guide](../fundamentals/block-composition.md)
- Browse the full architecture catalog: [Model Zoo Overview](../models/index.md)
