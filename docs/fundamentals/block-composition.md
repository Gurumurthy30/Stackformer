# Block Composition & Wiring

This guide details how `BlockConfig` drives the instantiation and internal wiring of `EncoderBlock`, `DecoderBlock`, `TransformerEncoder`, and `TransformerDecoder`.

---

## 1. The `BlockConfig` Dataclass

`BlockConfig` encapsulates all architectural choices for a Transformer layer into a single configuration object:

```python
from stackformer.modules import BlockConfig

cfg = BlockConfig(
    embed_dim=512,
    num_heads=8,
    hidden_dim=2048,
    attention="gqa_rope",
    num_kv_heads=2,
    ffn="swiglu",
    norm="rmsnorm",
    pre_norm=True,
    dropout=0.1,
    qkv_bias=True,
    eps=1e-5,
)
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `embed_dim` | `int` | *(Required)* | Hidden feature dimensionality $C$. Must be divisible by `num_heads`. |
| `num_heads` | `int` | *(Required)* | Number of query attention heads $H$. |
| `hidden_dim` | `int` | `0` | Inner feed-forward hidden dimension $M$. If `0`, automatically defaults to $4 \times \text{embed\_dim}$. |
| `attention` | `str` | `"mha"` | Attention variant: `"mha"`, `"mha_rope"`, `"mqa"`, `"mqa_rope"`, `"gqa"`, `"gqa_rope"`, `"self"`. |
| `num_kv_heads` | `int \| None` | `None` | Key/Value heads for GQA/MQA. Defaults to `num_heads` if `None`. |
| `ffn` | `str` | `"swiglu"` | FFN activation: `"relu"`, `"leakyrelu"`, `"gelu"`, `"sigmoid"`, `"silu"`, `"swiglu"`, `"geglu"`. |
| `norm` | `str` | `"rmsnorm"` | Normalization layer type: `"layernorm"`, `"rmsnorm"`. |
| `dropout` | `float` | `0.0` | Dropout probability across attention and FFN layers. |
| `pre_norm` | `bool` | `True` | Normalization placement: `True` for Pre-LN (modern standard), `False` for Post-LN (Vaswani original). |
| `mask_type` | `list` | `["causal"]` | Mask pattern list passed to attention modules. |
| `qkv_bias` | `bool` | `True` | Include learnable bias in Query, Key, and Value linear projections. |
| `eps` | `float` | `1e-5` | Numerical stability epsilon for normalization layers. |
| `device` | `str \| torch.device` | `"cpu"` | Target compute device for weight tensors. |
| `dtype` | `torch.dtype` | `torch.float32` | Tensor floating point precision. |

### Automatic Validation Rules

When `BlockConfig` is initialized, it validates:
1. **Hidden Dimension Default**: If `hidden_dim == 0`, it computes `hidden_dim = 4 * embed_dim`.
2. **Head Divisibility**: Raises `ValueError` if `embed_dim % num_heads != 0`.
3. **KV Head Divisibility**: Raises `ValueError` if `("gqa" in attention or "mqa" in attention)` and `num_heads % num_kv_heads != 0`.

---

## 2. `EncoderBlock` Internal Wiring

`EncoderBlock` connects self-attention and FFN sub-layers using residual connections:

### Pre-Norm Wiring (Modern Standard: GPT, LLaMA, Gemma)

Normalizes inputs *before* each sub-layer, preserving a clean identity residual highway:

```text
x ───┬───▶ [ Norm 1 ] ───▶ [ Self-Attention ] ───▶ (+) ───┬───▶ [ Norm 2 ] ───▶ [ FFN ] ───▶ (+) ───▶ Output
     │                                              ▲      │                                   ▲
     └──────────────────────────────────────────────┘      └───────────────────────────────────┘
```

```python
# Pre-Norm computation
x = x + self.self_attn(self.norm1(x), mask=mask)
x = x + self.ffn(self.norm2(x))
```

### Post-Norm Wiring (Vaswani Transformer Original)

Normalizes activations *after* residual addition:

```text
x ───┬───▶ [ Self-Attention ] ───▶ (+) ───▶ [ Norm 1 ] ───┬───▶ [ FFN ] ───▶ (+) ───▶ [ Norm 2 ] ───▶ Output
     │                              ▲                      │                   ▲
     └──────────────────────────────┘                      └───────────────────┘
```

```python
# Post-Norm computation
x = self.norm1(x + self.self_attn(x, mask=mask))
x = self.norm2(x + self.ffn(x))
```

---

## 3. `DecoderBlock` Internal Wiring

`DecoderBlock` incorporates cross-attention to attend over encoder memory:

```mermaid
graph TD
    subgraph DecoderBlock Wiring (Pre-Norm)
        X["Target Input x (B, T, C)"] --> N1["Norm 1"]
        N1 --> SA["Self-Attention (Causal)"]
        SA --> Add1["(+) Residual 1"]
        X --> Add1

        Add1 --> N2["Norm 2"]
        N2 --> CA["Cross-Attention"]
        Mem["Encoder Memory (B, S, C)"] --> CA
        CA --> Add2["(+) Residual 2"]
        Add1 --> Add2

        Add2 --> N3["Norm 3"]
        N3 --> FFN["Feed-Forward Network"]
        FFN --> Add3["(+) Residual 3"]
        Add2 --> Add3

        Add3 --> Out["Decoder Block Output (B, T, C)"]
    end
```

---

## 4. `TransformerEncoder` & `TransformerDecoder`

Backbones chain multiple blocks into a deep hierarchy:

```python
from stackformer.modules import BlockConfig, TransformerEncoder, TransformerDecoder

cfg = BlockConfig(embed_dim=256, num_heads=4, ffn="gelu", norm="layernorm")

# 1. 6-layer Encoder stack with sinusoidal positional embeddings
encoder = TransformerEncoder(cfg, num_layers=6, pos_embedding="sinusoidal")

# 2. 6-layer Decoder stack
decoder = TransformerDecoder(cfg, num_layers=6, pos_embedding="sinusoidal")

src = torch.randn(2, 20, 256)
tgt = torch.randn(2, 15, 256)

memory = encoder(src)            # (2, 20, 256)
out = decoder(tgt, memory)       # (2, 15, 256)
```

---

## 5. Backward Compatibility: `TransformerBlock`

For older codebases that expect single block initialization with individual keyword arguments, StackFormer provides `TransformerBlock` as an alias that automatically instantiates an underlying `EncoderBlock(cfg)`:

```python
from stackformer.modules.layer import TransformerBlock

# Traditional keyword initialization
block = TransformerBlock(
    embed_dim=256,
    num_heads=4,
    hidden_dim=1024,
    attention="mha",
    ffn="relu",
    norm="layernorm",
)
y = block(torch.randn(2, 32, 256))
```

