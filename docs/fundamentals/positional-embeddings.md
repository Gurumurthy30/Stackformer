# Positional Embeddings

Self-attention is an inherently permutation-invariant operation: swapping two token positions produces identical attention scores unless sequence position information is explicitly introduced.

StackFormer supports three positional representation strategies:
1. **Sinusoidal Positional Embeddings**: Fixed deterministic trigonometric encoding (Vaswani et al., 2017).
2. **Absolute Learned Positional Embeddings**: Parameter lookup table learned via backpropagation (Devlin et al., 2018; Radford et al., 2019).
3. **Rotary Position Embeddings (RoPE)**: Relative position rotation applied directly to attention Query/Key vectors (Su et al., 2021).

---

## Comparison of Strategies

| Embedding Strategy | Type | Learnable? | Max Sequence Extrapolation | Integration Point | Used In |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sinusoidal** | Absolute | No (Fixed) | Good (smooth harmonic extrapolation) | Input token representation | Vaswani Transformer |
| **Absolute Learned** | Absolute | Yes | Poor (hard limit at `seq_len`) | Input token representation | GPT-1/2, BERT, RoBERTa, ViT |
| **RoPE** | Relative | No (Frequencies fixed) | Excellent | Inside Attention (on Q & K) | LLaMA-1/2, Gemma-1, Mistral |

---

## 1. Sinusoidal Positional Embeddings

### Mathematical Formulation

Given position $pos \in [0, T-1]$ and channel index $i \in [0, \frac{C}{2}-1]$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/C}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/C}}\right)$$

This allows the model to learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

### Implementation

```python
import torch
from stackformer.modules import SinusoidalPositionalEmbedding

# 1. Instantiate module
pos_emb = SinusoidalPositionalEmbedding(max_seq_len=512, embed_dim=256)

# 2. Add to token representations
x = torch.randn(2, 32, 256)
x_pos = x + pos_emb(x)

print("Output shape:", x_pos.shape)  # torch.Size([2, 32, 256])
```

---

## 2. Absolute Learned Positional Embeddings

### Mechanism

Maintains a learnable weight tensor $W_{pos} \in \mathbb{R}^{T_{\text{max}} \times C}$. For a sequence of length $T$, slices the first $T$ positions:

$$x_{\text{out}} = x + W_{pos}[:T, :]$$

### Implementation

```python
import torch
from stackformer.modules import AbsolutePositionEmbedding

# 1. Instantiate module with learnable embedding table
learned_pos = AbsolutePositionEmbedding(max_seq_len=512, embed_dim=256)

# 2. Add to token representations
x = torch.randn(2, 32, 256)
x_pos = x + learned_pos(x)

print("Output shape:", x_pos.shape)  # torch.Size([2, 32, 256])
```

---

## 3. Rotary Position Embeddings (RoPE)

### Mechanism

Instead of adding an offset vector to the input embeddings, RoPE rotates the Query and Key projections in 2D coordinate pairs by angles proportional to their sequence position:

$$\theta_i = 10000^{-2(i-1)/d}$$

For token position $m$, Query vector $q$ is rotated by $m \theta$:

$$\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix} \mapsto \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

### Important Usage Guideline

> [!IMPORTANT]
> When using RoPE, positional encodings are computed **inside** the attention sub-layer. Do **not** pass a `pos_embedding` argument to `TransformerEncoder` or `TransformerDecoder` when `attention="mha_rope"`, `"gqa_rope"`, or `"mqa_rope"`. Doing so would apply dual positional encodings and degrade attention scores.

```python
from stackformer.modules import BlockConfig, TransformerEncoder

# Correct: RoPE handled internally by attention
cfg = BlockConfig(embed_dim=512, num_heads=8, attention="gqa_rope")
encoder = TransformerEncoder(cfg, num_layers=4, pos_embedding=None)
```

