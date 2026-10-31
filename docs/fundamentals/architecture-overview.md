# StackFormer Architecture Overview

StackFormer structures Transformer development into four clean, hierarchical abstraction tiers:

```mermaid
graph TD
    subgraph Tier 4: Applications & Models
        M1["GPT-1 / GPT-2"]
        M2["LLaMA-1 / LLaMA-2"]
        M3["Gemma-1 (2B/7B)"]
        M4["BERT / RoBERTa"]
        M5["Transformer (Seq2Seq)"]
        M6["ViT & SegFormer"]
    end

    subgraph Tier 3: Stacks & Backbones
        TE["TransformerEncoder"]
        TD["TransformerDecoder"]
    end

    subgraph Tier 2: Blocks
        EB["EncoderBlock"]
        DB["DecoderBlock"]
    end

    subgraph Tier 1: Orthogonal Primitives
        Attn["Attention (MHA, GQA, MQA, RoPE)"]
        FFN["Feed-Forward (GELU, SwiGLU, GeGLU)"]
        Norm["Normalization (LayerNorm, RMSNorm)"]
        Pos["Positional (Sinusoidal, Absolute, RoPE)"]
    end

    Attn --> EB
    FFN --> EB
    Norm --> EB
    
    Attn --> DB
    FFN --> DB
    Norm --> DB

    EB --> TE
    DB --> TD

    TE --> M1
    TE --> M2
    TE --> M3
    TE --> M4
    TE --> M6
    TD --> M5
```

---

## 1. Tier 1: Orthogonal Primitives

At the lowest level, `stackformer.modules` houses orthogonal building blocks:
- **Attention Modules**: Calculate token-to-token contextual representations.
- **Feed-Forward Networks (FFN)**: Apply point-wise non-linear feature projections across channels.
- **Normalization**: Rescale activations across hidden dimensions to maintain stable gradients.
- **Positional Embeddings**: Inject sequence position awareness into permutation-invariant attention matrices.

Each primitive is implemented with consistent tensor shape expectations:
- Sequence input: `(Batch, Sequence, EmbedDim)` or `(B, T, C)`.
- Attention heads: `(Batch, Heads, Sequence, HeadDim)` or `(B, H, T, D)`.

---

## 2. Tier 2: Blocks (`EncoderBlock` & `DecoderBlock`)

A Transformer block wires attention, feed-forward, and normalization sub-layers together with residual connections.

StackFormer supports two primary block designs:
- **`EncoderBlock`**: Consists of self-attention followed by an FFN, each surrounded by residual additions and normalization. Used in encoder models (BERT, ViT) and causal decoder models (GPT, LLaMA, Gemma).
- **`DecoderBlock`**: Consists of causal self-attention, cross-attention (attending over external memory from an encoder), and an FFN. Used in sequence-to-sequence models (Vaswani Transformer).

---

## 3. Tier 3: Stacks (`TransformerEncoder` & `TransformerDecoder`)

Backbone stacks chain $N$ identical blocks together:
- Accept a single `BlockConfig` and instantiate $N$ layers via `nn.ModuleList`.
- Optionally apply initial positional embeddings (`sinusoidal` or `absolute`) at the input.
- Optionally apply a final normalization layer before emitting hidden states.

```python
from stackformer.modules import BlockConfig, TransformerEncoder

cfg = BlockConfig(embed_dim=512, num_heads=8, ffn="swiglu", norm="rmsnorm")
encoder = TransformerEncoder(cfg, num_layers=12)
```

---

## 4. Tier 4: Model Zoo

The top tier bundles backbones with task-specific embeddings and projection heads:
- Causal Language Models: Embeddings + Encoder Stack (with causal mask) + `lm_head` linear projection.
- Masked Language Models: Embeddings (Token + Position + Segment) + Encoder Stack + MLM projection head.
- Vision Models: Patch Embedding Conv/Linear + Encoder Stack + Classifier head.
- Sequence-to-Sequence: Source Embedding + `TransformerEncoder` + Target Embedding + `TransformerDecoder` + Generation head.

---

## 5. Model Lifecycle & `BaseModel`

All foundation and vision models in StackFormer inherit from `BaseModel`, establishing a clean separation of concerns for large-scale model workflows:

```mermaid
graph TD
    M["Model(config)<br><i>Define architecture and register modules</i>"]
    CW["create_weights()<br><i>Initialization strategies</i>"]
    LW["load_weights()<br><i>Checkpoint loading</i>"]
    SW["save_weights()<br><i>Checkpoint saving (.pt / .safetensors)</i>"]
    MAT["materialize()<br><i>Meta-device to allocated memory</i>"]
    DP["Distributed Preparation<br><i>FSDP / DDP / Device placement</i>"]
    OPT["Optimizer + Training<br><i>Training lifecycle</i>"]

    M --> CW
    M --> LW
    CW --> SW
    LW --> SW
    M --> MAT
    MAT --> CW
    MAT --> LW
    SW --> DP
    CW --> DP
    LW --> DP
    DP --> OPT
```

### Standard Model Lifecycle Methods

| Method | Role | Description |
| :--- | :--- | :--- |
| `__init__(config)` | Architecture definition | Declares submodules and backbone blocks without requiring immediate GPU memory. |
| `create_weights(method="default")` | Weight initialization | Initializes parameters using architecture defaults or mathematical strategies (`xavier`, `kaiming`, `normal`, `orthogonal`). |
| `materialize(device="cpu")` | Memory allocation | Converts meta-device tensors (`torch.device("meta")`) into physical allocated memory on target compute devices. |
| `load_weights(path=None)` | Checkpoint loading | Loads weights from local `.pt` or `.safetensors` files, or fetches canonical pretrained model weights. |
| `save_weights(path)` | Checkpoint saving | Serializes state dictionaries in PyTorch or SafeTensors format with duplicate memory protection. |
| `parameter_count(trainable_only=False)` | Parameter accounting | Returns the total or trainable parameter count across all submodules. |

### Example Workflow

```python
import torch
from stackformer.config import ModelConfig
from stackformer.models import GPT2

config = ModelConfig(
    vocab_size=50257,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    seq_len=1024,
    hidden_dim=3072,
)

# 1. Instantiate on meta-device (zero RAM allocation)
with torch.device("meta"):
    model = GPT2(config)

# 2. Materialize onto compute device
model.materialize(device="cpu")

# 3. Initialize weights with Xavier or architecture default
model.create_weights(method="default")

# 4. Save and load checkpoints (.pt or .safetensors)
model.save_weights("checkpoints/gpt2.safetensors")
model.load_weights("checkpoints/gpt2.safetensors")

# 5. Inspect parameter count
print(f"Total parameters: {model.parameter_count():,}")
```


