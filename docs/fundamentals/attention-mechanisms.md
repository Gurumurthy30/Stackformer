# Attention Mechanisms

Attention mechanisms allow Transformer models to weigh token relationships dynamically across sequence positions.

StackFormer supports 10 distinct attention module variants:
- **Multi-Head Attention (MHA)**: Standard multi-head architecture (Vaswani et al., 2017).
- **Grouped-Query Attention (GQA)**: Query heads grouped into shared Key/Value clusters (Ainslie et al., 2023).
- **Multi-Query Attention (MQA)**: All query heads share a single Key/Value head (Shazeer, 2019).
- **Cross-Attention**: Cross-sequence attention attending over external encoder memory.
- **RoPE Variants**: Rotary Position Embeddings computed directly on Query/Key head representations.
- **Stateful KV-Cache Modules**: Persistent key-value buffer attention for autoregressive inference.

---

## 1. Comparing MHA, GQA, and MQA

```mermaid
graph TD
    subgraph Multi-Head Attention (MHA)
        Q1["Q Head 1"] --> KV1["KV Head 1"]
        Q2["Q Head 2"] --> KV2["KV Head 2"]
        Q3["Q Head 3"] --> KV3["KV Head 3"]
        Q4["Q Head 4"] --> KV4["KV Head 4"]
    end

    subgraph Grouped-Query Attention (GQA)
        GQ1["Q Head 1"] --> GKV1["Shared KV Head 1"]
        GQ2["Q Head 2"] --> GKV1
        GQ3["Q Head 3"] --> GKV2["Shared KV Head 2"]
        GQ4["Q Head 4"] --> GKV2
    end

    subgraph Multi-Query Attention (MQA)
        MQ1["Q Head 1"] --> MKV["Single Shared KV Head"]
        MQ2["Q Head 2"] --> MKV
        MQ3["Q Head 3"] --> MKV
        MQ4["Q Head 4"] --> MKV
    end
```

| Dimension | Multi-Head Attention (MHA) | Grouped-Query Attention (GQA) | Multi-Query Attention (MQA) |
| :--- | :--- | :--- | :--- |
| **Number of KV Heads** | $H_{KV} = H_Q$ | $1 < H_{KV} < H_Q$ | $H_{KV} = 1$ |
| **KV Cache Memory** | $100\%$ baseline | $12.5\% - 25\%$ of baseline | $\frac{1}{H_Q}$ of baseline |
| **Inference Bandwidth** | High memory bandwidth bound | Balanced memory bandwidth | Minimal memory bandwidth |
| **Quality Retention** | Full capacity | Near-identical to MHA | Slight degradation on complex tasks |
| **Used In** | GPT-2, BERT, RoBERTa | LLaMA-2, Gemma-1 7B | Gemma-1 2B, PaLM |

---

## 2. Mathematical Formulation

Standard scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

Where:
- $Q \in \mathbb{R}^{B \times H_Q \times T_q \times d_k}$ (Query tensor)
- $K \in \mathbb{R}^{B \times H_{KV} \times T_k \times d_k}$ (Key tensor)
- $V \in \mathbb{R}^{B \times H_{KV} \times T_k \times d_v}$ (Value tensor)
- $M \in \mathbb{R}^{T_q \times T_k}$ (Attention mask: $0$ for valid positions, $-\infty$ for masked positions)
- $d_k = \frac{\text{embed\_dim}}{H_Q}$ (Per-head dimensionality)

In GQA and MQA, Keys and Values are expanded along head dimensions via `torch.repeat_interleave` to match query head counts:

$$\text{group\_size} = \frac{H_Q}{H_{KV}}$$

---

## 3. Multi-Head Attention (`MultiHeadAttention`)

### Constructor & Forward Signatures

```python
from stackformer.modules import MultiHeadAttention

attn = MultiHeadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    qkv_bias=True,
    mask_type=["causal"],
)
```

- **Input Tensor**: $x \in \mathbb{R}^{B \times T \times C}$
- **Mask Flag**: `mask=True` enforces lower-triangular causal masking (no future token leakage).
- **Output Tensor**: $y \in \mathbb{R}^{B \times T \times C}$

### Code Example

```python
import torch
from stackformer.modules import MultiHeadAttention

mha = MultiHeadAttention(embed_dim=256, num_heads=4)
x = torch.randn(2, 16, 256)
y = mha(x, mask=True)
print("MHA output shape:", y.shape)  # torch.Size([2, 16, 256])
```

---

## 4. Grouped-Query Attention (`GroupQueryAttention`)

### Constructor & Forward Signatures

```python
from stackformer.modules import GroupQueryAttention

gqa = GroupQueryAttention(
    embed_dim=512,
    num_heads=8,
    num_kv_heads=2,  # 4 query heads per KV head
    dropout=0.0,
    qkv_bias=False,
)
```

- **Constraint**: `num_heads % num_kv_heads == 0`.
- **KV Projection Size**: Instead of projecting $K$ and $V$ to `embed_dim`, GQA projects them to `num_kv_heads * head_dim`. For 8 query heads and 2 KV heads, KV projection size is reduced by 75%.

### Code Example

```python
import torch
from stackformer.modules import GroupQueryAttention

gqa = GroupQueryAttention(embed_dim=512, num_heads=8, num_kv_heads=2)
x = torch.randn(2, 32, 512)
out = gqa(x, mask=True)
print("GQA output shape:", out.shape)  # torch.Size([2, 32, 512])
```

---

## 5. Multi-Query Attention (`MultiQueryAttention`)

All query heads share a single key head and a single value head:

```python
from stackformer.modules import MultiQueryAttention

mqa = MultiQueryAttention(embed_dim=512, num_heads=8)
x = torch.randn(2, 32, 512)
out = mqa(x, mask=True)
print("MQA output shape:", out.shape)  # torch.Size([2, 32, 512])
```

---

## 6. Rotary Position Embeddings (RoPE) Variants

Standard attention is permutation invariant. RoPE rotates Query and Key representations in complex 2D planes according to sequence index:

$$\tilde{q}_m = R_{\Theta, m} q_m, \quad \tilde{k}_n = R_{\Theta, n} k_n$$

Such that:

$$\langle \tilde{q}_m, \tilde{k}_n \rangle = \text{Re}(q_m k_n^* e^{i(m-n)\theta})$$

The dot product depends strictly on relative distance $(m - n)$.

StackFormer provides built-in RoPE variants:
- `MultiHeadAttentionWithRoPE` (`attention="mha_rope"`)
- `GroupQueryAttentionWithRoPE` (`attention="gqa_rope"`)
- `MultiQueryAttentionWithRoPE` (`attention="mqa_rope"`)

### Usage with BlockConfig

```python
from stackformer.modules import BlockConfig, EncoderBlock

cfg = BlockConfig(
    embed_dim=512,
    num_heads=8,
    num_kv_heads=2,
    attention="gqa_rope",
    ffn="swiglu",
    norm="rmsnorm",
)
block = EncoderBlock(cfg)
x = torch.randn(2, 64, 512)
y = block(x, mask=True)
print("RoPE Block output shape:", y.shape)  # torch.Size([2, 64, 512])
```

---

## 7. Cross-Attention (`CrossMultiHeadAttention`)

Used in encoder-decoder architectures (such as `Transformer` and `DecoderBlock`) where queries originate from the decoder target sequence, while keys and values originate from encoder memory:

```python
from stackformer.modules import CrossMultiHeadAttention

cross_attn = CrossMultiHeadAttention(embed_dim=256, num_heads=4)

target = torch.randn(2, 10, 256)   # Decoder sequence (B, T, C)
memory = torch.randn(2, 25, 256)   # Encoder sequence (B, S, C)

# Q from target, K/V from memory
out = cross_attn(x=target, memory=memory)
print("Cross-attention output shape:", out.shape)  # torch.Size([2, 10, 256])
```

---

## 8. Scaled Dot-Product Execution Engine

StackFormer's attention engine (`stackformer.modules.attention_engine`) automatically selects the optimal execution path:
- **`_run_sdpa`**: Calls PyTorch's native `F.scaled_dot_product_attention`, automatically dispatching to FlashAttention-2 or xFormers memory-efficient kernels on CUDA.
- **`_run_flex_attention`**: Dispatches to PyTorch 2.5+ FlexAttention when custom block masks or document packing masks are supplied.
- **Causal Mask Dispatch**: Dynamically constructs causal lower-triangular boolean masks matching batch and sequence lengths.

