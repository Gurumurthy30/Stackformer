# Practical Examples

This section contains self-contained, end-to-end Python examples demonstrating how to use StackFormer for common research and production workflows.

Each example is complete and can be run directly.

---

## Example Directory

| Guide | Description | Key Modules Used |
| :--- | :--- | :--- |
| [**Custom Transformer Architecture**](custom-transformer.md) | Construct a custom Transformer combining GQA, RoPE, SwiGLU, and RMSNorm using `BlockConfig`. | `stackformer.modules.layer`, `BlockConfig` |
| [**Language Model Training**](language-model-training.md) | Train a causal language model with `Trainer`, mixed precision (AMP), cosine schedule, and SafeTensors checkpointing. | `stackformer.models.GPT2`, `stackformer.engine.Trainer` |
| [**Autoregressive Text Generation**](text-generation.md) | Generate sequences with greedy, temperature, top-k, and nucleus (top-p) sampling. | `stackformer.generate.text_generate`, `GenerationConfig` |
| [**Distributed Training with DDP**](distributed-training.md) | Launch multi-GPU training with PyTorch's `torchrun` and StackFormer's DDP engine. | `stackformer.distributed`, `Trainer` |

---

## Running Examples Locally

All examples can be executed directly from the root of the repository:

```bash
# Language model training
python -m examples.simple_trainer

# Distributed training (single or multi-process)
torchrun --standalone --nproc_per_node=2 examples/train_ddp.py
```

