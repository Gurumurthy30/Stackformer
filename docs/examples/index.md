# Practical Examples

This section contains self-contained, end-to-end Python examples demonstrating how to use StackFormer for modular architecture construction and sequence inference.

Each example is complete and can be run directly.

---

## Example Directory

| Guide | Description | Key Modules Used |
| :--- | :--- | :--- |
| [**Custom Transformer Architecture**](custom-transformer.md) | Construct a custom Transformer combining GQA, RoPE, SwiGLU, and RMSNorm using `BlockConfig`. | `stackformer.modules.layer`, `BlockConfig` |
| [**Autoregressive Text Generation**](text-generation.md) | Generate sequences with greedy, temperature, top-k, and nucleus (top-p) sampling. | `stackformer.generate.text_generate`, `GenerationConfig` |

---

## Running Examples

You can run these examples directly in your Python environment or script:

```python
import torch
from stackformer.models import GPT2
from stackformer.generate import text_generate

model = GPT2(vocab_size=1000, num_layers=4, embed_dim=256, num_heads=4, seq_len=128)
prompt = torch.tensor([[1, 2, 3, 4]])
out = text_generate(model, prompt, max_new_tokens=10)
print(out)
```
