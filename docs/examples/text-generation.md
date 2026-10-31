# Example: Autoregressive Text Generation

This example demonstrates how to perform sequence generation using `text_generate()` and compare sampling configurations (greedy decoding vs. temperature vs. nucleus top-p sampling).

---

## Complete Runnable Script

```python
import torch
from stackformer.models import GPT2
from stackformer.generate import text_generate
from stackformer.config import GenerationConfig

def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_size = 1000
    seq_len = 128

    # 1. Initialize Model
    model = GPT2(
        vocab_size=vocab_size,
        num_layers=4,
        embed_dim=256,
        num_heads=4,
        seq_len=seq_len,
    ).to(device)
    model.eval()

    # 2. Define Prompt IDs
    # E.g. 5 initial prompt tokens
    prompt = torch.tensor([[12, 450, 89, 230, 45]], device=device)
    print("Initial prompt tokens:", prompt.tolist()[0])

    # 3. Strategy A: Greedy Decoding (temperature -> 0, or lowest temp)
    print("\n--- Strategy A: Deterministic Sampling (Low Temp) ---")
    greedy_config = GenerationConfig(
        max_context_len=128,
        max_new_tokens=20,
        temperature=0.01,
        top_k=1,
    )
    with torch.no_grad():
        out_greedy = text_generate(model, prompt, generation_config=greedy_config)
    print("Generated (Deterministic):", out_greedy.tolist()[0])

    # 4. Strategy B: Top-K Sampling
    print("\n--- Strategy B: Top-K Sampling (k=10, temp=0.8) ---")
    topk_config = GenerationConfig(
        max_context_len=128,
        max_new_tokens=20,
        temperature=0.8,
        top_k=10,
    )
    with torch.no_grad():
        out_topk = text_generate(model, prompt, generation_config=topk_config)
    print("Generated (Top-K):", out_topk.tolist()[0])

    # 5. Strategy C: Nucleus (Top-P) Sampling
    print("\n--- Strategy C: Nucleus Top-P Sampling (p=0.9, temp=0.7) ---")
    topp_config = GenerationConfig(
        max_context_len=128,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9,
    )
    with torch.no_grad():
        out_topp = text_generate(model, prompt, generation_config=topp_config)
    print("Generated (Top-P):", out_topp.tolist()[0])

    # 6. Strategy D: Early Stopping on EOS Token (e.g. EOS token ID = 2)
    print("\n--- Strategy D: Early Stopping on EOS Token ---")
    eos_config = GenerationConfig(
        max_context_len=128,
        max_new_tokens=50,
        temperature=0.8,
        eos_token_id=2,
    )
    with torch.no_grad():
        out_eos = text_generate(model, prompt, generation_config=eos_config)
    print("Generated (with EOS):", out_eos.tolist()[0])


if __name__ == "__main__":
    main()
```

---

## Sampling Behavior Notes

- **Deterministic (Low Temp)**: Always selects the token with the highest predicted score. Output is 100% reproducible for a given input.
- **Top-K (e.g. $K=10$)**: Truncates candidate choices to the 10 highest-probability tokens.
- **Nucleus Top-P (e.g. $P=0.90$)**: Dynamically expands or shrinks the candidate token set so the sum of probabilities is $\ge 90\%$.

