# StackFormer Model Architecture Zoo

StackFormer ships production-faithful PyTorch implementations of 10 canonical language and vision Transformer architectures.

Every model in the zoo is built directly on top of StackFormer's modular primitives (`stackformer.modules`), demonstrating that state-of-the-art models can be expressed declaratively with clean abstractions.

---

## Architectural Comparison Matrix

| Model | Task / Domain | Attention Mechanism | Positional Embedding | Normalization | FFN Activation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [**GPT-1 & GPT-2**](gpt.md) | Autoregressive Causal LM | MHA (Causal) | Absolute Learned | Post-LN (GPT-1)<br>Pre-LN (GPT-2) | GELU |
| [**LLaMA-1 & LLaMA-2**](llama.md) | Autoregressive Causal LM | MHA + RoPE (LLaMA-1)<br>GQA + RoPE (LLaMA-2) | RoPE | Pre-RMSNorm | SwiGLU |
| [**Gemma-1 (2B & 7B)**](gemma.md) | Autoregressive Causal LM | MQA + RoPE (2B)<br>GQA + RoPE (7B) | RoPE | Pre-RMSNorm | GeGLU |
| [**BERT**](bert.md) | Masked LM / Bidirectional | MHA (Bidirectional) | Absolute Learned + Segment | Post-LayerNorm | GELU |
| [**RoBERTa**](roberta.md) | Masked LM / Bidirectional | MHA (Bidirectional) | Absolute Learned (Offset) | Post-LayerNorm | GELU |
| [**Vaswani Transformer**](transformer.md) | Seq2Seq Translation | MHA + Cross-MHA | Sinusoidal Fixed | Post-LayerNorm | ReLU |
| [**Vision Transformer (ViT)**](vit.md) | Image Classification | MHA (Bidirectional) | Absolute Learned | Pre-LayerNorm | GELU |
| [**SegFormer (B0–B5)**](segformer.md) | Semantic Segmentation | Spatial Reduction Attn | Mix-FFN (Conv-embedded) | Pre-LayerNorm | GELU |

---

## Pretrained Weights Disclaimer

> [!IMPORTANT]
> StackFormer provides **native, verifiable PyTorch architecture definitions**, not a hosted distribution repository of billions of pretrained model checkpoint weights.
> 
> You can instantiate, train, fine-tune, study, and export any model architecture in this zoo with zero external dependencies beyond pure PyTorch.

