# Example: Distributed Multi-GPU Training (DDP)

This example demonstrates how to set up and run distributed data-parallel (DDP) training across multiple processes using StackFormer's `Trainer` and PyTorch's `torchrun` launcher.

---

## Complete Training Script (`train_ddp_example.py`)

Save the following file as `train_ddp_example.py`:

```python
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stackformer.models import GPT2
from stackformer.engine import Trainer
from stackformer.distributed import cleanup_distributed, get_rank, is_main_process

def main():
    # 1. Prepare synthetic dataset
    num_samples = 512
    seq_len = 64
    vocab_size = 1000
    batch_size = 8

    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.randint(0, vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size)

    # 2. Instantiate Model
    model = GPT2(
        vocab_size=vocab_size,
        num_layers=4,
        embed_dim=128,
        num_heads=4,
        seq_len=seq_len,
    )

    # 3. Configure Trainer with DDP enabled
    # The Trainer automatically calls init_distributed(), partitions data with
    # DistributedSampler, and wraps the model with DistributedDataParallel.
    trainer = Trainer(
        model=model,
        train_dataloader=loader,
        use_ddp=True,
        ddp_backend="nccl" if torch.cuda.is_available() else "gloo",
        device="auto",
        max_epochs=2,
        lr=1e-3,
        grad_accumulation_step=2,
        checkpoint_dir="checkpoints/ddp_run",
    )

    if is_main_process():
        print(f"Starting distributed training across ranks...")

    # 4. Run Training
    trainer.fit()

    if is_main_process():
        print("Distributed training finished.")

    # 5. Clean up process group
    cleanup_distributed()


if __name__ == "__main__":
    main()
```

---

## Launch Commands

### Single-Node Multi-GPU (e.g. 4 GPUs)

Launch with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 train_ddp_example.py
```

### CPU Multi-Process Testing

You can test distributed training even without a GPU by launching multiple CPU workers:

```bash
torchrun --standalone --nproc_per_node=2 train_ddp_example.py
```

### Multi-Node Multi-GPU Cluster

On Node 0 (Master node with IP `192.168.1.10`):
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.10" \
    --master_port=29500 \
    train_ddp_example.py
```

On Node 1 (Worker node):
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.10" \
    --master_port=29500 \
    train_ddp_example.py
```

---

## Key DDP Invariants Handled by StackFormer

1. **Rank 0 Logging**: Console logging and file writes are guarded with `is_main_process()` to avoid garbled logs.
2. **Safe Checkpoint Serialization**: In multi-GPU runs, weights are gathered and written to disk only by Rank 0, preventing file collisions.
3. **Epoch Shuffling**: In multi-epoch training, the random seed of `DistributedSampler` is updated each epoch so each rank receives a new non-overlapping shuffle.

