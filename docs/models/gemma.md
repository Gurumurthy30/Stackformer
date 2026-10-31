# Gemma-1 (2B & 7B) Architectures

StackFormer provides native implementations of Google's **Gemma-1 2B** and **Gemma-1 7B** open language models (Gemma Team, Google DeepMind, 2024).

Implemented in `stackformer.models.gemma` and exposed at the top level as `Gemma1_2B`, `Gemma1_7B`, `Gemma_1_2B`, `Gemma_1_7B`, `gemma_1_2b`, and `gemma_1_7b`.

---

## 1. Architecture Overview

Gemma is a decoder-only language model built on Google's Gemini architecture research.

Key architectural innovations include:
1. **Embedding Scaling**: Token embeddings are normalized and multiplied by $\sqrt{C}$ ($\sqrt{\text{embed\_dim}}$) before entering the Transformer layers.
2. **GeGLU Activations**: Feed-forward sub-layers use Gated Linear Units with Gaussian Error Linear Unit activations ($\text{GeGLU}$), providing smooth gradient flow.
3. **RMSNorm with Unit Offset**: Normalizes activations using RMSNorm where weights are parameterized with a $+1$ offset ($y = \frac{x}{\text{RMS}(x)} \odot (1 + \gamma)$).
4. **Attention Sizing**:
   - **Gemma-1 2B**: Uses **Multi-Query Attention (MQA)** ($H_{KV} = 1$) to maximize memory efficiency on consumer and mobile hardware.
   - **Gemma-1 7B**: Uses **Grouped-Query Attention (GQA)** with 16 Key/Value heads.

---

## 2. Constructor Parameters

```python
Gemma1_2B(
    vocab_size: int,
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    seq_len: int,
    dropout: float = 0.0,
    hidden_dim: int = 0,
    eps: float = 1e-5,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
)

Gemma1_7B(
    vocab_size: int,
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    seq_len: int,
    dropout: float = 0.0,
    hidden_dim: int = 0,
    eps: float = 1e-5,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
)
```

- `vocab_size` (int): Gemma standard vocabulary size (typically 256,000 for Gemma sentencepiece tokenizer).
- `embed_dim` (int): Feature dimensionality (e.g. 2048 for 2B, 3072 for 7B).
- `hidden_dim` (int, default=0): Inner GeGLU hidden dimension. When `0`, computed via `hidden_dim_helper` or $4 \times \text{embed\_dim}$.

---

## 3. Initialization & Forward Pass

### Shapes
- **Input `input_ids`**: Integer token tensor of shape `(Batch, Sequence)` where values $\in [0, \text{vocab\_size}-1]$.
- **Output `logits`**: Floating point tensor of shape `(Batch, Sequence, VocabSize)`.

```python
import torch
from stackformer.models import Gemma1_2B

# 1. Instantiate a compact Gemma-1 2B model configuration
model = Gemma1_2B(
    vocab_size=32000,
    num_layers=4,
    embed_dim=256,
    num_heads=4,
    seq_len=128,
)

# 2. Input batch: Batch=2, Sequence=16
input_ids = torch.randint(0, 32000, (2, 16))

# 3. Compute forward logits
logits = model(input_ids)
print("Gemma logits shape:", logits.shape)  # torch.Size([2, 16, 32000])
```

---

## 4. Training with PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stackformer.models import Gemma1_2B

# 1. Synthetic training batch
inputs = torch.randint(0, 5000, (32, 16))
targets = inputs.clone()
loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

# 2. Initialize model, optimizer, and loss criterion
model = Gemma1_2B(
    vocab_size=5000,
    num_layers=2,
    embed_dim=128,
    num_heads=4,
    seq_len=64,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

# 3. Execute training loop
model.train()
for epoch in range(1):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits.view(-1, 5000), batch_y.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item():.4f}")
```

---

## 5. Implementation Limitations

1. **Vocabulary Scaling**: Gemma uses a massive 256,000 token vocabulary in full production. On memory-constrained devices, ensure `vocab_size` is adjusted to your actual tokenizer size to avoid large `lm_head` weight matrices ($256000 \times C$).
2. **Weight Pretraining**: Checkpoint weights must be trained natively or loaded via SafeTensors.

