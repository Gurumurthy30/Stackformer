# Trainer

The `Trainer` class is StackFormer's high-level training orchestrator. It coordinates device placement, optimizer and learning rate scheduler construction, gradient accumulation, automatic mixed precision (AMP), DistributedDataParallel (DDP), validation intervals, and checkpoint management into a single, cohesive training workflow.

---

## Constructor Parameters

```python
from stackformer.engine import Trainer

trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=None,                 # Auto-created if None
    scheduler=None,                 # Auto-created if None
    criterion=None,                 # Defaults to language_modeling_cross_entropy
    device="auto",                  # "auto", "cuda", or "cpu"
    seed=42,
    max_epochs=10,
    max_train_steps=None,           # Optional step limit
    max_eval_steps=None,            # Optional eval step limit
    eval_every_n_epochs=1,
    save_every_n_epochs=1,
    grad_accumulation_step=1,
    max_grad_norm=1.0,              # Gradient clipping norm
    checkpoint_dir="checkpoints",
    resume_from=None,               # E.g. "latest" or "best"
    monitor=None,                   # Custom Logger or MetricTracker
    use_amp=False,
    use_ddp=False,
    ddp_backend=None,               # "nccl" or "gloo"
    lr=3e-4,
    weight_decay=0.01,
    optimizer_name="adamw",         # "adamw", "adam", "sgd", "rmsprop"
    scheduler_name="cosine",        # "cosine", "linear", "step", "none"
    warmup_steps=100,
    sharded_checkpoint=False,       # Save/load via PyTorch DCP
)
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | *required* | The PyTorch neural network to train. |
| `train_dataloader` | `DataLoader \| None` | `None` | DataLoader providing training batches `(inputs, targets)` or `dict`. |
| `val_dataloader` | `DataLoader \| None` | `None` | Optional DataLoader for validation evaluation. |
| `optimizer` | `Optimizer \| None` | `None` | Custom optimizer instance. If `None`, instantiated using `optimizer_name`, `lr`, and `weight_decay`. |
| `scheduler` | `LRScheduler \| None` | `None` | Custom scheduler. If `None`, instantiated using `scheduler_name`, `warmup_steps`, and total steps. |
| `criterion` | `Callable \| None` | `None` | Loss calculation function `(logits, targets) -> loss`. Defaults to `language_modeling_cross_entropy`. |
| `device` | `str` | `"auto"` | Target compute device (`"auto"` selects CUDA if available, otherwise CPU). |
| `seed` | `int \| None` | `None` | Random seed for deterministic reproducibility across NumPy, Python, and PyTorch. |
| `max_epochs` | `int` | `1` | Total number of full dataset passes. |
| `max_train_steps` | `int \| None` | `None` | Global step limit. Training stops early if this step count is reached. |
| `max_eval_steps` | `int \| None` | `None` | Maximum evaluation steps to run per validation cycle. |
| `eval_every_n_epochs` | `int` | `1` | Evaluation frequency in epochs. |
| `save_every_n_epochs` | `int` | `1` | Checkpointing frequency in epochs. |
| `grad_accumulation_step` | `int` | `1` | Number of micro-batches to accumulate before optimizer step. |
| `max_grad_norm` | `float \| None` | `None` | Maximum gradient norm threshold for `torch.nn.utils.clip_grad_norm_`. |
| `checkpoint_dir` | `str` | `"checkpoints"`| Output directory for saved SafeTensors weights and training state. |
| `resume_from` | `str \| None` | `None` | Checkpoint name to restore at initialization (e.g. `"latest"`). |
| `monitor` | `Any \| None` | `None` | Logging monitor or `Logger` instance for metric streaming. |
| `use_amp` | `bool` | `False` | Enable automatic mixed precision (CUDA/MPS). |
| `use_ddp` | `bool` | `False` | Wrap model with `DistributedDataParallel`. |
| `ddp_backend` | `str \| None` | `None` | PyTorch distributed backend (`"nccl"` on Linux GPU, `"gloo"` on Windows or CPU). |
| `lr` | `float` | `3e-4` | Learning rate for auto-created optimizer. |
| `weight_decay` | `float` | `0.01` | Weight decay penalty applied to eligible weight matrices. |
| `optimizer_name` | `str` | `"adamw"` | Name of optimizer to instantiate (`"adamw"`, `"adam"`, `"sgd"`, `"rmsprop"`). |
| `scheduler_name` | `str` | `"none"` | Name of scheduler (`"cosine"`, `"linear"`, `"step"`, `"plateau"`, `"none"`). |
| `warmup_steps` | `int` | `0` | Number of initial linear warmup steps for learning rate scheduler. |
| `sharded_checkpoint`| `bool` | `False` | Enable Distributed Checkpoint (DCP) sharded state saving for large models. |

---

## Core Methods

### `.fit()`

Executes the main training loop across all configured epochs:

```python
trainer.fit()
```

During execution, `.fit()`:
1. Synchronizes seeds across workers.
2. Iterates over epochs $1 \dots \text{max\_epochs}$.
3. Dispatches dataloader batches through `Engine.train_one_epoch()`.
4. Evaluates on `val_dataloader` when `epoch % eval_every_n_epochs == 0`.
5. Automatically saves checkpoints to `checkpoint_dir` according to `save_every_n_epochs`.

### `.evaluate()`

Manually runs an evaluation pass over `val_dataloader`:

```python
metrics = trainer.evaluate()
print(f"Validation Loss: {metrics.get('val_loss'):.4f}")
```

### `.save_checkpoint(name)` and `.load_checkpoint(name)`

Manually persist or restore training state:

```python
# Save current model and optimizer state
trainer.save_checkpoint(name="checkpoint_epoch_5")

# Restore state
trainer.load_checkpoint(name="checkpoint_epoch_5")
```

---

## Complete Example: Training GPT-2 with Custom Settings

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from stackformer.models import GPT2
from stackformer.engine import Trainer
from stackformer.logging import Logger

# 1. Prepare Model and Dataset
model = GPT2(
    vocab_size=50257,
    num_layers=6,
    embed_dim=384,
    num_heads=6,
    seq_len=256,
)

# Mock tokenized text dataset
data = torch.randint(0, 50257, (128, 256))
train_loader = DataLoader(TensorDataset(data), batch_size=8, shuffle=True)
val_loader = DataLoader(TensorDataset(data[:16]), batch_size=8)

# 2. Setup Logger
logger = Logger(csv=True, experiment_name="gpt2_training_run")

# 3. Instantiate Trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    max_epochs=5,
    lr=3e-4,
    weight_decay=0.01,
    optimizer_name="adamw",
    scheduler_name="cosine",
    warmup_steps=50,
    grad_accumulation_step=2,
    max_grad_norm=1.0,
    use_amp=torch.cuda.is_available(),
    monitor=logger,
    checkpoint_dir="checkpoints/gpt2_run",
)

# 4. Start Training
trainer.fit()
```

