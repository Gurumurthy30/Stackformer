# Example: Language Model Training

This example demonstrates how to set up an end-to-end language modeling training pipeline using StackFormer's `GPT2` model and `Trainer` with:
- Synthetic tokenized text dataset
- AdamW optimizer with cosine learning rate schedule and linear warmup
- Automatic weight-decay parameter grouping
- Automatic mixed precision (AMP)
- SafeTensors checkpointing

---

## Complete Runnable Script

```python
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from stackformer.models import GPT2
from stackformer.engine import Trainer
from stackformer.logging import Logger

def generate_synthetic_lm_dataset(num_samples: int, seq_len: int, vocab_size: int):
    """Generate synthetic autoregressive training dataset (inputs, targets)."""
    # Tokens: (num_samples, seq_len + 1)
    tokens = torch.randint(0, vocab_size, (num_samples, seq_len + 1))
    
    # Inputs: tokens[:-1], Targets: tokens[1:]
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    return TensorDataset(inputs, targets)

def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running training on device: {device}")

    vocab_size = 5000
    seq_len = 128
    batch_size = 8

    # 1. Create Datasets and DataLoaders
    train_dataset = generate_synthetic_lm_dataset(num_samples=256, seq_len=seq_len, vocab_size=vocab_size)
    val_dataset = generate_synthetic_lm_dataset(num_samples=64, seq_len=seq_len, vocab_size=vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model
    model = GPT2(
        vocab_size=vocab_size,
        num_layers=4,
        embed_dim=256,
        num_heads=4,
        seq_len=seq_len,
        dropout=0.1,
    )

    # 3. Setup Logger
    logger = Logger(
        csv=True,
        tensorboard=False,
        log_dir="experiments/logs",
        experiment_name="gpt2_synthetic_run",
    )

    # 4. Configure Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        max_epochs=3,
        lr=5e-4,
        weight_decay=0.01,
        optimizer_name="adamw",
        scheduler_name="cosine",
        warmup_steps=20,
        grad_accumulation_step=2,
        max_grad_norm=1.0,
        use_amp=(device == "cuda"),
        eval_every_n_epochs=1,
        save_every_n_epochs=1,
        checkpoint_dir="checkpoints/gpt2_run",
        monitor=logger,
    )

    # 5. Execute Training Loop
    print("Starting training with StackFormer Trainer...")
    trainer.fit()
    print("Training successfully completed.")

    # 6. Evaluate
    val_metrics = trainer.evaluate()
    print("Final Validation Metrics:", val_metrics)


if __name__ == "__main__":
    main()
```

---

## Output Artifacts

Upon completion of training, the following structure will be saved in your working directory:

```text
checkpoints/gpt2_run/
├── epoch_1.safetensors    # Model weights for epoch 1
├── epoch_1.pt             # Optimizer & state for epoch 1
├── latest.safetensors     # Final weights
└── latest.pt              # Final optimizer state

experiments/logs/gpt2_synthetic_run/
└── metrics.csv            # Step-by-step loss and LR metrics
```

