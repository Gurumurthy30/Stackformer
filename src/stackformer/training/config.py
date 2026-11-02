"""Configuration dataclass and validation for Stackformer training."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable
import torch


@dataclass
class TrainConfig:
    """Configuration for model training in Stackformer.

    Parameters:
        max_epochs: Maximum number of epochs to train.
        max_steps: Maximum number of optimizer steps (if set, stops when hit).
        lr: Learning rate.
        weight_decay: Weight decay factor for optimizer.
        optimizer: Optimizer name ("adamw", "adam", "sgd").
        scheduler: LR scheduler name ("cosine", "linear", "none").
        warmup_steps: Warmup steps for scheduler.
        loss: Loss function name or callable fn(outputs, batch) -> scalar.
        parallel: Parallelism strategy ("none", "ddp", "fsdp2").
        tp_degree: Tensor parallel degree.
        pp_degree: Pipeline parallel degree.
        cp_degree: Context parallel degree.
        mixed_precision: Precision mode ("no", "fp16", "bf16").
        activation_checkpointing: AC strategy ("none", "selective", "full").
        compile: Whether to torch.compile the model blocks.
        grad_accum_steps: Number of micro-batches to accumulate gradients.
        max_grad_norm: Max gradient norm for clipping (None to disable).
        device: Device to use ("auto", "cuda", "mps", "cpu").
        seed: Random seed for reproducibility.
        eval_every_n_epochs: Evaluate model every N epochs.
        eval_every_n_steps: Evaluate model every N optimizer steps.
        max_eval_steps: Maximum batches to evaluate per evaluation run.
        save_every_n_epochs: Save checkpoint every N epochs.
        save_every_n_steps: Save checkpoint every N optimizer steps.
        checkpoint_dir: Directory to save checkpoints.
        resume_from: Checkpoint directory path or "auto" to resume.
        save_best: Whether to track and save the best model.
        monitor: Metric name to monitor for best model.
        mode: Best metric mode ("min" or "max").
        keep_last_n: Number of checkpoints to keep (prunes older).
        async_checkpoint: Whether to save checkpoints asynchronously.
    """

    max_epochs: int | None = 1
    max_steps: int | None = None
    lr: float = 3e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 0
    loss: str | Callable[[Any, Any], torch.Tensor] = "cross_entropy"
    loss_mode: str = "auto"
    parallel: str = "none"
    tp_degree: int = 1
    pp_degree: int = 1
    cp_degree: int = 1
    mixed_precision: str = "no"
    activation_checkpointing: str = "none"
    compile: bool = False
    grad_accum_steps: int = 1
    max_grad_norm: float | None = 1.0
    device: str = "auto"
    seed: int = 42
    eval_every_n_epochs: int | None = 1
    eval_every_n_steps: int | None = None
    max_eval_steps: int | None = None
    save_every_n_epochs: int | None = 1
    save_every_n_steps: int | None = None
    checkpoint_dir: str = "./checkpoints"
    resume_from: str | None = None
    save_best: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    keep_last_n: int = 3
    async_checkpoint: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration values and raise ValueError on invalid settings."""
        if self.max_epochs is None and self.max_steps is None:
            raise ValueError("At least one of 'max_epochs' or 'max_steps' must be specified.")
        if self.max_epochs is not None and self.max_epochs <= 0:
            raise ValueError(f"'max_epochs' must be positive, got {self.max_epochs}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"'max_steps' must be positive, got {self.max_steps}")
        if self.lr <= 0:
            raise ValueError(f"'lr' must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"'weight_decay' must be non-negative, got {self.weight_decay}")
        if self.grad_accum_steps < 1:
            raise ValueError(f"'grad_accum_steps' must be >= 1, got {self.grad_accum_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"'warmup_steps' must be >= 0, got {self.warmup_steps}")
        if self.loss_mode not in ("auto", "lm", "classification"):
            raise ValueError(
                f"'loss_mode' must be one of ('auto', 'lm', 'classification'), got '{self.loss_mode}'"
            )

        valid_optimizers = {"adamw", "adam", "sgd"}
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(
                f"Unsupported optimizer '{self.optimizer}'. Supported: {sorted(valid_optimizers)}"
            )

        valid_schedulers = {"cosine", "linear", "none", ""}
        if self.scheduler.lower() not in valid_schedulers:
            raise ValueError(
                f"Unsupported scheduler '{self.scheduler}'. Supported: {sorted(valid_schedulers)}"
            )

        valid_precision = {"no", "fp16", "bf16"}
        if self.mixed_precision.lower() not in valid_precision:
            raise ValueError(
                f"Unsupported mixed_precision '{self.mixed_precision}'. Supported: {sorted(valid_precision)}"
            )

        valid_parallel = {"none", "ddp", "fsdp2"}
        if self.parallel.lower() not in valid_parallel:
            raise ValueError(
                f"Unsupported parallel mode '{self.parallel}'. Supported: {sorted(valid_parallel)}"
            )

        valid_ac = {"none", "selective", "full"}
        if self.activation_checkpointing.lower() not in valid_ac:
            raise ValueError(
                f"Unsupported activation_checkpointing '{self.activation_checkpointing}'. Supported: {sorted(valid_ac)}"
            )

        if self.mode.lower() not in {"min", "max"}:
            raise ValueError(f"'mode' must be 'min' or 'max', got {self.mode}")

    def resolve_device(self) -> torch.device:
        """Resolve 'auto' device to torch.device (cuda > mps > cpu)."""
        dev_str = self.device.lower().strip()
        if dev_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(dev_str)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        # Loss might be callable, keep as string or repr
        if callable(self.loss):
            d["loss"] = getattr(self.loss, "__name__", repr(self.loss))
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        """Construct TrainConfig from a dictionary, mapping legacy aliases."""
        d = copy.deepcopy(data)
        # Map aliases
        aliases = {
            "max_train_steps": "max_steps",
            "total_steps": "max_steps",
            "max_eval_step": "max_eval_steps",
            "grad_accumulation_step": "grad_accum_steps",
            "optimizer_name": "optimizer",
            "scheduler_name": "scheduler",
            "criterion": "loss",
            "training_config": None,
        }
        for alias, target in aliases.items():
            if alias in d:
                val = d.pop(alias)
                if target is not None and target not in d:
                    d[target] = val

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

