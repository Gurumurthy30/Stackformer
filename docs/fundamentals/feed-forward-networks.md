# Feed-Forward Networks (FFN) & Gated Activations

Feed-Forward Networks (FFNs) process each token representation independently and identically across sequence positions, providing non-linear channel-mixing capacity that complements the sequence-mixing behavior of self-attention.

StackFormer supports 7 FFN activation architectures categorized into **Standard Two-Layer MLPs** and **Three-Layer Gated Linear Units (GLU)**.

---

## 1. Supported FFN Architectures

| FFN Type | Flag in `BlockConfig` | Layers | Canonical Class | Primary Use Cases |
| :--- | :--- | :--- | :--- | :--- |
| **GELU** | `ffn="gelu"` | 2 Linear | `FeedForwardGELU` (`FF_GELU`) | GPT-1/2, BERT, RoBERTa, ViT |
| **SwiGLU** | `ffn="swiglu"` | 3 Linear (Gated) | `FeedForwardSwiGLU` (`FF_SwiGLU`) | LLaMA-1/2, Mistral, DeepSeek |
| **GeGLU** | `ffn="geglu"` | 3 Linear (Gated) | `FeedForwardGeGLU` (`FF_GeGLU`) | Gemma-1, PaLM |
| **ReLU** | `ffn="relu"` | 2 Linear | `FeedForwardReLU` (`FF_ReLU`) | Vaswani Transformer (2017) |
| **SiLU** | `ffn="silu"` | 2 Linear | `FeedForwardSiLU` (`FF_SiLU`) | Modern lightweight backbones |
| **LeakyReLU**| `ffn="leakyrelu"` | 2 Linear | `FeedForwardLeakyReLU` (`FF_LeakyReLU`) | Custom research networks |
| **Sigmoid** | `ffn="sigmoid"` | 2 Linear | `FeedForwardSigmoid` (`FF_Sigmoid`) | Gate weighting networks |

---

## 2. Standard MLPs vs Gated Linear Units (GLU)

```mermaid
graph TD
    subgraph Standard MLP (e.g. GELU / ReLU)
        In1["Input x (B, T, C)"] --> W1["Linear 1 (C -> M)"]
        W1 --> Act1["Activation: GELU(x W1)"]
        Act1 --> W2["Linear 2 (M -> C)"]
        W2 --> Out1["Output (B, T, C)"]
    end

    subgraph Gated Linear Unit (e.g. SwiGLU / GeGLU)
        In2["Input x (B, T, C)"] --> Gate["Gate Linear (C -> M)"]
        In2 --> Up["Up Linear (C -> M)"]
        Gate --> Act2["Swish / GELU"]
        Act2 --> Mult["Element-wise Multiplication (⊗)"]
        Up --> Mult
        Mult --> Down["Down Linear (M -> C)"]
        Down --> Out2["Output (B, T, C)"]
    end
```

### Standard Two-Layer MLP Formulation

$$\text{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2$$

Where $W_1 \in \mathbb{R}^{C \times M}$ projects up to hidden dimension $M$, and $W_2 \in \mathbb{R}^{M \times C}$ projects back to embedding dimension $C$.

### Gated Linear Unit Formulation (SwiGLU & GeGLU)

Introduced by Shazeer (2020), Gated Linear Units replace the single non-linear projection with a bilinear product between a non-linearly activated gate and a linear projection:

$$\text{GLU}(x, W, V, b, c) = \sigma(x W + b) \otimes (x V + c)$$

- **SwiGLU** (using $\text{Swish}(x) = x \cdot \text{sigmoid}(\beta x)$):
  $$\text{SwiGLU}(x) = (\text{Swish}(x W_{\text{gate}}) \otimes x W_{\text{up}}) W_{\text{down}}$$

- **GeGLU** (using $\text{GELU}(x)$):
  $$\text{GeGLU}(x) = (\text{GELU}(x W_{\text{gate}}) \otimes x W_{\text{up}}) W_{\text{down}}$$

---

## 3. Hidden Dimension Sizing ($M$)

In standard MLPs, the inner dimension is conventionally four times the embedding size:

$$M = 4 \times C$$

In Gated Linear Units (SwiGLU/GeGLU), because three projection matrices are used instead of two, setting $M = 4 \times C$ would increase total model parameter count by 50%. To maintain equal parameter counts with standard MLPs, models like LLaMA set:

$$M \approx \frac{8}{3} \times C$$

StackFormer provides the `hidden_dim_helper` utility to compute this automatically:

```python
from stackformer.utils import hidden_dim_helper

# Computes 2/3 * 4 * embed_dim rounded to multiples of 256 for tensor cores
hidden_dim = hidden_dim_helper(embed_dim=512)
print("Computed SwiGLU hidden dim:", hidden_dim)  # 1365 rounded
```

---

## 4. Code Examples

### Direct Module Usage

```python
import torch
from stackformer.modules import FeedForwardGELU, FeedForwardSwiGLU

x = torch.randn(2, 16, 512)

# Standard GELU FFN
gelu_ffn = FeedForwardGELU(embed_dim=512, hidden_dim=2048, dropout=0.1)
y_gelu = gelu_ffn(x)
print("GELU FFN output shape:", y_gelu.shape)      # torch.Size([2, 16, 512])

# SwiGLU Gated FFN
swiglu_ffn = FeedForwardSwiGLU(embed_dim=512, hidden_dim=1376, dropout=0.0)
y_swiglu = swiglu_ffn(x)
print("SwiGLU FFN output shape:", y_swiglu.shape)  # torch.Size([2, 16, 512])
```

### Specifying via `BlockConfig`

```python
from stackformer.modules import BlockConfig, EncoderBlock

# Configure SwiGLU block
cfg = BlockConfig(
    embed_dim=512,
    num_heads=8,
    hidden_dim=1376,
    ffn="swiglu",      # Options: "relu", "leakyrelu", "gelu", "sigmoid", "silu", "swiglu", "geglu"
    attention="gqa_rope",
    norm="rmsnorm",
)
block = EncoderBlock(cfg)
```

