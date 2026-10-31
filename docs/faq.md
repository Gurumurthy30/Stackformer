# Frequently Asked Questions (FAQ)

Frequently asked questions regarding StackFormer's scope, design philosophy, features, and usage.

---

### What is StackFormer's primary focus?

StackFormer is a **modular Transformer architecture framework** designed for architecture experimentation, research prototyping, and custom model building. Its primary focus is **declarative block composability**: allowing you to swap attention variants, feed-forward activations, normalization schemes, and positional embeddings in seconds via `BlockConfig` without rewriting monolithic model files.

---

### Does StackFormer provide pretrained model weights?

**No.** StackFormer provides clean, canonical, production-faithful PyTorch **architecture implementations** (e.g. GPT-2, LLaMA-2, Gemma-1, ViT, SegFormer), not a hosted hub of thousands of pretrained checkpoint weights. If you need a drop-in download for billions of open-source weights with pre-packaged tokenizers, use Hugging Face `transformers`. If you need to build, study, train, modify, or extend Transformer architectures with zero abstraction bloat, use StackFormer.

---

### How does StackFormer compare to nanoGPT?

| Dimension | nanoGPT | StackFormer |
| :--- | :--- | :--- |
| **Scope** | Minimal single-file GPT-2 training script | Complete modular framework across language and vision |
| **Model Zoo** | GPT-2 only | GPT-1/2, LLaMA-1/2, Gemma-1, BERT, RoBERTa, Vaswani Transformer, ViT, SegFormer |
| **Composability** | Hardcoded attention and MLP blocks | Declarative `BlockConfig` (10 attention variants, 7 FFNs, 2 norms) |
| **Training Engine** | Custom loop in `train.py` | Standard PyTorch `nn.Module` compatible with any trainer or framework |
| **Inference** | Simple top-k script | Stateful KV-cache accelerated `text_generate()`, `prefill()`, and `decode()` |

---

### Can I use `torch.compile` with StackFormer?

**Yes.** StackFormer is implemented in pure PyTorch using standard `nn.Module` and functional operations (`F.scaled_dot_product_attention`, `torch.matmul`). There are no Python-C boundary breaks or custom opaque C++ extensions in the forward path, enabling clean compilation via:

```python
import torch
from stackformer.models import GPT2

model = GPT2(vocab_size=10000, num_layers=4, embed_dim=256, num_heads=4, seq_len=128)
compiled_model = torch.compile(model)
```

---

### What is the difference between `TransformerEncoder` and `EncoderBlock`?

- `EncoderBlock` represents **a single Transformer layer** containing one self-attention module, one feed-forward module, residual connections, and two normalization layers.
- `TransformerEncoder` is a **stack of $N$ `EncoderBlock` layers**, plus an optional initial positional embedding module (`sinusoidal` or `absolute`) and an optional final normalization layer.

---

### Why does RoPE not need a separate positional embedding in `TransformerEncoder`?

Rotary Position Embeddings (RoPE) are applied **directly to Query and Key tensors inside the attention mechanism** during the attention computation, rather than being added to the token embeddings at the input of the Transformer stack.

When using `attention="mha_rope"`, `"gqa_rope"`, or `"mqa_rope"`, leave `pos_embedding=None` on `TransformerEncoder`. Adding a separate positional embedding would double-encode position and corrupt the attention geometry.

---

### How does StackFormer handle checkpoints?

StackFormer's `CheckpointManager` stores model weights in the **SafeTensors (`.safetensors`)** format, eliminating the arbitrary code execution vulnerabilities of Python `pickle`. In addition, it writes an accompanying metadata JSON file preserving optimizer states, learning rate scheduler progress, training epoch counters, and hyperparameter configurations. For distributed workloads, it supports PyTorch Distributed Checkpoint (DCP).

---

### What attention backends are used?

StackFormer's attention engine routes scaled dot-product attention queries through PyTorch's native `F.scaled_dot_product_attention` (SDPA), which automatically dispatches to **FlashAttention-2**, **Memory-Efficient Attention (xFormers)**, or high-precision C++ math kernels depending on hardware and tensor shapes.

