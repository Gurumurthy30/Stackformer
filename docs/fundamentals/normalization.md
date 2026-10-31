# Normalization Layers & Pre-LN vs Post-LN

Normalization layers rescale intermediate activations across channel dimensions, dampening internal covariate shift and ensuring stable gradient propagation during backpropagation.

StackFormer supports two normalization implementations:
- **`LayerNorm`**: Standard Layer Normalization with mean centering and variance scaling (Ba et al., 2016).
- **`RMSNorm`**: Root Mean Square Normalization with scale-only invariance (Zhang & Sennrich, 2019).

---

## 1. LayerNorm vs RMSNorm

### Layer Normalization (`LayerNorm`)

LayerNorm computes the mean $\mu$ and variance $\sigma^2$ across the feature dimension $C$:

$$\mu = \frac{1}{C} \sum_{i=1}^C x_i, \quad \sigma^2 = \frac{1}{C} \sum_{i=1}^C (x_i - \mu)^2$$

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

Where:
- $\gamma \in \mathbb{R}^C$ is a learnable scale parameter (initialized to 1).
- $\beta \in \mathbb{R}^C$ is a learnable shift/bias parameter (initialized to 0).
- $\epsilon > 0$ prevents division by zero.

### Root Mean Square Normalization (`RMSNorm`)

RMSNorm hypothesizes that the scaling property of LayerNorm is what stabilizes training, while mean-centering ($\mu$) can be discarded without loss of performance:

$$\text{RMS}(x) = \sqrt{\frac{1}{C} \sum_{i=1}^C x_i^2 + \epsilon}$$

$$y = \frac{x}{\text{RMS}(x)} \odot \gamma$$

**Key Advantages of RMSNorm**:
1. **Computational Efficiency**: Discards mean computation and subtraction, saving 10%–50% of normalization kernel runtime on GPUs.
2. **Reduced Parameters**: Eliminates the $\beta$ bias parameter.
3. **Standard in Modern LLMs**: Adopted by LLaMA, LLaMA-2, Gemma, Mistral, and DeepSeek.

---

## 2. Pre-Norm vs Post-Norm Stability

StackFormer allows selecting between Pre-Norm and Post-Norm via the `pre_norm` boolean flag in `BlockConfig`.

```mermaid
graph TD
    subgraph Pre-Norm (Modern LLM Standard)
        x1["Input x"] --> Norm1["Norm"]
        Norm1 --> Sub1["Attention / FFN"]
        Sub1 --> Add1["(+) Residual"]
        x1 --> Add1
        Add1 --> Out1["Output (Identity Highway Preserved)"]
    end

    subgraph Post-Norm (Vaswani Original)
        x2["Input x"] --> Sub2["Attention / FFN"]
        Sub2 --> Add2["(+) Residual"]
        x2 --> Add2
        Add2 --> Norm2["Norm"]
        Norm2 --> Out2["Output"]
    end
```

| Characteristic | Pre-Norm (`pre_norm=True`) | Post-Norm (`pre_norm=False`) |
| :--- | :--- | :--- |
| **Residual Highway** | Pure identity: gradients flow unimpeded from output to input | Scaled by normalization at each layer |
| **Training Stability** | Extremely stable; trains deep networks ($N > 32$) easily | Prone to gradient vanishing/explosion in early epochs |
| **Warmup Requirement** | Minimal warmup needed | Strict linear warmup essential to avoid divergence |
| **Final Layer Output** | Requires a final norm before output projection | Does not require a final norm |
| **Used In** | GPT-2, LLaMA-1/2, Gemma-1, ViT | Vaswani Transformer (2017), BERT, RoBERTa |

---

## 3. Code Examples

### Direct Module Instantiation

```python
import torch
from stackformer.modules import LayerNorm, RMSNorm

x = torch.randn(2, 16, 512)

# Standard LayerNorm
ln = LayerNorm(embed_dim=512, eps=1e-5)
y_ln = ln(x)
print("LayerNorm output shape:", y_ln.shape)  # torch.Size([2, 16, 512])

# RMSNorm
rms = RMSNorm(embed_dim=512, eps=1e-5)
y_rms = rms(x)
print("RMSNorm output shape:", y_rms.shape)   # torch.Size([2, 16, 512])
```

### Specifying via `BlockConfig`

```python
from stackformer.modules import BlockConfig, EncoderBlock

# Configure modern Pre-RMSNorm block
cfg = BlockConfig(
    embed_dim=512,
    num_heads=8,
    norm="rmsnorm",   # Options: "layernorm", "rmsnorm"
    pre_norm=True,    # Options: True (Pre-LN), False (Post-LN)
    eps=1e-6,
)

block = EncoderBlock(cfg)
```

