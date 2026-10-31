# Text Generation

The primary inference interface in StackFormer is `text_generate()`, located in `stackformer.generate`. It provides autoregressive decoding with automatic KV-cache detection, context window capping, early stopping on end-of-sequence (EOS) tokens, and batched generation.

---

## The `text_generate()` Function

### Signature & Parameters

```python
from stackformer.generate import text_generate

output_ids = text_generate(
    model=model,
    prompt_ids=prompt_ids,
    max_context_len=128,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    top_p=1.0,
    eos_token_id=None,
    generation_config=None,
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | *required* | Language model instance. |
| `prompt_ids` | `torch.Tensor` | *required* | Prompt sequence tensor of shape `(B, T)` or 1D `(T,)`. |
| `max_context_len` | `int` | `128` | Maximum context window length. Contexts exceeding this are right-truncated during decoding. |
| `max_new_tokens` | `int` | `50` | Maximum number of new tokens to generate. |
| `temperature` | `float` | `1.0` | Sampling temperature ($T > 0$). Lower values produce more deterministic text; higher values increase randomness. |
| `top_k` | `int \| None` | `None` | Filters logits to top-$k$ highest probability tokens before sampling. |
| `top_p` | `float` | `1.0` | Filters logits to top-$p$ cumulative probability mass (Nucleus sampling). |
| `eos_token_id` | `int \| None` | `None` | Stop token ID. If produced, halts generation for that sequence. |
| `generation_config` | `GenerationConfig \| None` | `None` | Optional configuration dataclass overriding individual arguments. |

---

## Stopping on End-of-Sequence (`eos_token_id`)

When `eos_token_id` is supplied:
1. As each sequence in a batch produces the EOS token, it is marked as `finished`.
2. Completed sequences have future positions padded with `eos_token_id`.
3. If all sequences in the batch have completed, the generation loop halts early immediately without executing unnecessary future forward passes.

```python
# Stop generating as soon as token ID 2 is emitted
generated = text_generate(
    model=model,
    prompt_ids=prompt,
    max_new_tokens=100,
    eos_token_id=2,
)
```

---

## Using `GenerationConfig`

Instead of passing individual parameters, you can bundle decoding hyperparameters into a `GenerationConfig`:

```python
from stackformer.config import GenerationConfig
from stackformer.generate import text_generate

gen_cfg = GenerationConfig(
    max_context_len=256,
    max_new_tokens=60,
    temperature=0.7,
    top_k=40,
    top_p=0.95,
    eos_token_id=50256,
)

output = text_generate(
    model=model,
    prompt_ids=prompt,
    generation_config=gen_cfg,
)
```

---

## Batched Text Generation

`text_generate()` natively handles 2D batch tensors `(B, T)`:

```python
import torch
from stackformer.models import GPT2
from stackformer.generate import text_generate

model = GPT2(vocab_size=1000, num_layers=4, embed_dim=256, num_heads=4, seq_len=128)
model.eval()

# Batch of 3 prompts with 8 tokens each
batch_prompts = torch.randint(10, 900, (3, 8))

# Generates 15 new tokens per prompt in parallel
with torch.no_grad():
    outputs = text_generate(
        model=model,
        prompt_ids=batch_prompts,
        max_new_tokens=15,
        temperature=0.9,
    )

print("Output batch shape:", outputs.shape)  # torch.Size([3, 23])
```

