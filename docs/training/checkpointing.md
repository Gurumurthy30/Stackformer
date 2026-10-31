# Checkpointing & Model Serialization

StackFormer provides a dual-tiered checkpointing system via `CheckpointManager` in `stackformer.engine.checkpoint`:
1. **Consolidated SafeTensors Format**: Saves model weights into `.safetensors` files for zero-copy, secure model exchange, while keeping mutable optimizer, scheduler, and training progress metadata in a companion `.pt` file.
2. **Distributed Checkpointing (DCP)**: Supports high-performance sharded saving directly from GPU memory shards for distributed multi-GPU clusters.

---

## Checkpoint File Layout

A standard consolidated checkpoint produces two distinct files inside `checkpoint_dir`:

```text
checkpoints/
├── latest.safetensors    # Clean model state dict (SafeTensors format)
├── latest.pt             # Optimizer, scheduler, scaler, and training metadata
├── best.safetensors
└── best.pt
```

### Why SafeTensors for Weights?
- **Security**: Eliminates the arbitrary code execution vulnerabilities associated with Python's `pickle`.
- **Zero-Copy Loading**: Memory maps weights directly into host memory for near-instant model loading.
- **Interoperability**: Compatible with Hugging Face Hub, vLLM, and ONNX pipelines.

---

## `CheckpointManager` API

### Constructor Parameters

```python
from stackformer.engine import CheckpointManager

manager = CheckpointManager(
    output_dir="checkpoints/run_01",
    device="cpu",
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `output_dir` | `str \| Path` | *required* | Path to the directory where checkpoint files will be written. |
| `device` | `str \| torch.device` | `"cpu"` | Compute device on which state dictionaries will be reconstructed upon loading. |

---

## Saving Checkpoints

```python
state = {
    "model": model,
    "optimizer": optimizer,
    "scheduler": scheduler,
    "scaler": scaler,
    "epoch": 5,
    "global_step": 2500,
    "config": {"lr": 1e-4},
}

# Save consolidated checkpoint
saved_paths = manager.save(state, name="epoch_5")
print(saved_paths)
# {
#     "weights": "checkpoints/run_01/epoch_5.safetensors",
#     "state": "checkpoints/run_01/epoch_5.pt"
# }
```

In distributed environments, `manager.save()` automatically aggregates parameters and only executes disk writes on **Rank 0**, avoiding file corruption.

---

## Restoring Checkpoints

To resume training or load weights for inference:

```python
state = {
    "model": model,
    "optimizer": optimizer,
    "scheduler": scheduler,
    "scaler": scaler,
}

metadata = manager.load(
    name="epoch_5",
    state=state,
    broadcast_from_rank0=False,
)

print(f"Restored from epoch {metadata['epoch']} at step {metadata['global_step']}")
```

---

## Sharded Distributed Checkpointing (DCP)

For large models distributed across multiple nodes, gathering full weights to rank 0 can cause host memory exhaustion. `CheckpointManager` provides sharded saving via `torch.distributed.checkpoint`:

```python
# Sharded save (each rank writes its own shard in parallel)
manager.save_sharded(
    model=ddp_model,
    optimizer=optimizer,
    name="dcp_step_5000",
)

# Sharded load (each rank reads only its local shard directly)
manager.load_sharded(
    model=ddp_model,
    optimizer=optimizer,
    name="dcp_step_5000",
)
```

---

## Resuming with `Trainer`

When using `Trainer`, pass the checkpoint identifier via `resume_from`:

```python
from stackformer.engine import Trainer

trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    checkpoint_dir="checkpoints/run_01",
    resume_from="latest",  # Automatically restores model, optimizer, scheduler, and epoch
    max_epochs=10,
)
trainer.fit()
```

