# Training and Checkpoint Management in Stackformer

Stackformer provides a high-level `Trainer` engine designed for scalable, reproducible, and verifiable transformer model training across single-process, multi-GPU (DDP), and sharded (FSDP2) setups.

---

## Checkpoint Architecture & Security

Stackformer stores training checkpoints in modular directories structured as follows:

```
checkpoint_dir/
├── step_00000100/
│   ├── model.safetensors       # Model weights stored in zero-copy, safe binary format
│   ├── state.pt                # Optimizer, LR scheduler, step metadata, and RNG states
│   └── trainer_config.json     # Serialized TrainConfig hyperparameters
└── latest                      # Plain-text pointer referencing the latest step directory
```

### Security Considerations

Stackformer intentionally separates weights from dynamic runtime states:

1. **`model.safetensors` (Safe Weights)**:
   - Model parameters and persistent buffers are saved in the Hugging Face `safetensors` format.
   - Safetensors files contain pure raw tensor buffers without executable Python bytecode or pickled objects. Loading weights via `stackformer.load_weights(model, path)` is immune to arbitrary code execution attacks.

2. **`state.pt` (Execution State & Trust Boundary)**:
   - Resuming training bit-identically requires restoring Python random states, NumPy RNG state dicts, PyTorch CPU/CUDA RNG generators, and arbitrary PyTorch optimizer states (e.g. AdamW momentum/variance buffers).
   - Because Python's `pickle` is required to deserialize arbitrary NumPy RNG states and nested optimizer structures, `state.pt` is restored with `torch.load(..., weights_only=False)`.

> [!WARNING]
> **Checkpoint Trust Trade-off**:
> Because `state.pt` contains pickled runtime state dictionaries, **you should only resume training (`Trainer(..., resume_from=...)`) from trusted checkpoint files created by yourself or verified collaborators**.
> Never resume training from an untrusted third-party `state.pt` file downloaded from the internet. When importing third-party weights, import only the `model.safetensors` file.

---

## Checkpoint Loading Workflows

### 1. Resuming Full Training

```python
from stackformer.training import Trainer, TrainConfig

config = TrainConfig(
    output_dir="./runs/experiment_1",
    resume_from="./runs/experiment_1/step_00001000",  # or pass latest directory
)
trainer = Trainer(model=model, config=config, train_loader=train_loader)
trainer.train()
```

### 2. Loading Weights for Inference / Evaluation

```python
import stackformer

# Directly loads weights from a safetensors file or a checkpoint directory:
stackformer.load_weights(model, "./runs/experiment_1/step_00001000")
```
