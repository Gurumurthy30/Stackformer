# Stackformer Training Infrastructure

The `stackformer.training` package provides a lightweight, modular training engine engineered specifically for transformer architectures, scaling from single-device prototyping to multi-node distributed training.

---

## 1. Core Architecture

```text
               ┌───────────────────────────────┐
               │          TrainConfig          │
               └───────────────┬───────────────┘
                               │
               ┌───────────────▼───────────────┐
               │            Trainer            │
               └───────┬───────────────┬───────┘
                       │               │
       ┌───────────────▼──────┐ ┌──────▼───────────────┐
       │   Transformation     │ │     CheckpointManager │
       │ (parallelize.py)     │ │ (checkpoint.py)       │
       │ AC -> Compile -> DP  │ │ Atomic single/DCP     │
       └──────────────────────┘ └───────────────────────┘
```

### Components
- **`Trainer`** (`trainer.py`): Streamlined $\le 285$-line coordinator driving training loops, optimization steps, metrics accumulation, and callback/event dispatching.
- **`TrainConfig`** (`config.py`): Strongly-typed dataclass encapsulating optimization hyperparameters, hardware settings, checkpoint paths, and verbosity modes.
- **`TrainState`** (`state.py`): Immutable state container tracking `epoch`, `global_step`, `step_in_epoch`, and `best_metric`.
- **`CheckpointManager`** (`checkpoint.py`): Atomic, durable persistence engine supporting single-file (`state.pt`, `model.safetensors`, `meta.json`) and PyTorch Distributed Checkpoint (DCP).
- **`DisplayManager`** (`display.py`): Terminal output engine supporting rich formatted tables/progress bars with automatic graceful fallback to plain text.
- **`launch()`** (`launcher.py`): Distributed launcher spawning multi-worker processes using `torch.multiprocessing.spawn`.

---

## 2. Transformation Pipeline Order

Stackformer enforces a strict transformation order in `stackformer.training.parallelize`:

$$\text{Activation Checkpointing (AC)} \longrightarrow \text{torch.compile} \longrightarrow \text{Data-Parallel Wrap (DDP / FSDP2)}$$

1. **Activation Checkpointing**: Applied to repeating transformer blocks retrieved via `model.get_sharding_blocks()`. Supports `selective` (checkpointing attention only) or `full` block checkpointing.
2. **Torch Compile**: Compiles sharding blocks prior to distributed wrapping, maximizing inductor fusion without distributed collective boundary conflicts.
3. **Distributed Sharding**: Wraps the compiled model in PyTorch Distributed Data Parallel (`DDP`) or Fully Sharded Data Parallel 2 (`FSDP2` via `torch.distributed.fsdp.fully_shard`).

---

## 3. Atomic Checkpointing & Resumption

Every checkpoint saved by `CheckpointManager` is written to a unique directory `step_<N>_<timestamp>` using an atomic temporary-directory rename:

```text
checkpoints/
├── latest -> step_100_20261103_063000/
├── best -> step_80_20261103_061500/
└── step_100_20261103_063000/
    ├── state.pt            # Model, optimizer, scheduler & RNG states
    ├── model.safetensors   # Standalone zero-copy weights for inference
    └── meta.json           # Step, epoch, step_in_epoch, best_metric, config
```

### Bit-Identical Mid-Epoch Resumption
When resuming mid-epoch with `TrainConfig(resume_from="auto")`:
1. `CheckpointManager` restores `TrainState.step_in_epoch`.
2. RNG seeds (`torch`, `torch.cuda`, `numpy`, `python`) are restored exactly.
3. The training loop fast-forwards through previously executed batches in the epoch. For bit-identical data order across mid-epoch resumes, pass a seeded `torch.Generator` to `DataLoader(..., generator=g)`.

---

## 4. Multi-Worker Distributed Launch

To train across multiple processes on a single machine:

```python
from stackformer.training import launch, Trainer, TrainConfig

def train_worker():
    # Model and loader creation
    model = create_model()
    loader = create_loader()
    cfg = TrainConfig(parallel="ddp", max_epochs=3)
    
    trainer = Trainer(model, cfg, train_loader=loader)
    trainer.train()

if __name__ == "__main__":
    launch(train_worker, num_processes=2, backend="gloo")
```

