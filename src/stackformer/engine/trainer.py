"""High-level Trainer API for model training, validation, and checkpoint management.

Provides `Trainer` class to orchestrate end-to-end training loops with DDP, AMP, gradient accumulation,
and checkpointing.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset

from stackformer.amp import initialize_scaler
from stackformer.config import TrainingConfig
from stackformer.distributed.ddp import init_distributed, is_distributed, wrap_model_ddp
from stackformer.engine.checkpoint import CheckpointManager
from stackformer.engine.engine import Engine
from stackformer.engine.state import TrainingState
from stackformer.logging import Logger
from stackformer.optim.factories import create_optimizer, create_scheduler
from stackformer.optim.loss_fn import language_modeling_cross_entropy
from stackformer.utils.device import get_device, print_device_info
from stackformer.utils.utils import is_main_process, print_once, seed_everything


class Trainer:
    """Configurable training loop orchestrator for single-GPU and distributed research workloads.

    Simple explanation:
        `Trainer` automates setup of optimizer, learning rate scheduler, device placement,
        gradient accumulation, automatic mixed precision (AMP), logging, and checkpointing.

    Constructor args:
        model (torch.nn.Module): Neural network model to train.
        train_dataloader (DataLoader | None, default=None): Training DataLoader.
        val_dataloader (DataLoader | None, default=None): Validation DataLoader.
        optimizer (torch.optim.Optimizer | None, default=None): Optional custom optimizer.
        scheduler (Any | None, default=None): Optional custom learning rate scheduler.
        criterion (Callable | None, default=None): Loss calculation callable (defaults to language_modeling_cross_entropy, targeting language modeling tasks).
        device (str, default="auto"): Compute device selection ("auto", "cpu", "cuda").
        seed (int | None, default=None): Random seed for reproducibility.
        max_epochs (int, default=1): Total number of training epochs.
        max_train_steps (int | None, default=None): Optional maximum step limit.
        max_eval_steps (int | None, default=None): Optional evaluation step limit.
        eval_every_n_epochs (int, default=1): Validation frequency in epochs.
        save_every_n_epochs (int, default=1): Checkpoint frequency in epochs.
        grad_accumulation_step (int, default=1): Gradient accumulation factor.
        max_grad_norm (float | None, default=None): Gradient norm clipping threshold.
        checkpoint_dir (str, default="checkpoints"): Directory path to store saved checkpoints.
        resume_from (str | None, default=None): Checkpoint name to restore at initialization.
        monitor (Any | None, default=None): Logging backend monitor instance.
        scaler (Any | None, default=None): Custom AMP GradScaler.
        use_amp (bool, default=False): Enable automatic mixed precision.
        use_ddp (bool, default=False): Enable DistributedDataParallel wrapping.
        ddp_backend (str | None, default=None): PyTorch distributed backend ("nccl", "gloo").
        lr (float, default=3e-4): Learning rate for auto-created optimizer.
        weight_decay (float, default=0.01): Weight decay parameter.
        optimizer_name (str, default="adamw"): Optimizer name ("adamw", "adam", "sgd").
        scheduler_name (str, default="none"): LR scheduler name ("cosine", "linear", "none").
        warmup_steps (int, default=0): Number of warmup steps for LR scheduler.
        total_steps (int | None, default=None): Total training steps for scheduler.
        max_steps (int | None, default=None): Backward-compatible alias for max_train_steps.
        max_eval_step (int | None, default=None): Backward-compatible alias for max_eval_steps.
        training_config (TrainingConfig | None, default=None): Optional TrainingConfig object.
        sharded_checkpoint (bool, default=False): Save/load via Distributed Checkpoint (DCP).
        broadcast_weights_from_rank0 (bool, default=False): Broadcast rank 0 weights on load.

    Example:
        >>> trainer = Trainer(model=model, train_dataloader=train_loader, max_epochs=5)
        >>> trainer.fit()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        max_epochs: int = 1,
        max_train_steps: Optional[int] = None,
        max_eval_steps: Optional[int] = None,
        eval_every_n_epochs: int = 1,
        save_every_n_epochs: int = 1,
        grad_accumulation_step: int = 1,
        max_grad_norm: Optional[float] = None,
        checkpoint_dir: str = "checkpoints",
        resume_from: Optional[str] = None,
        monitor: Optional[Any] = None,
        scaler: Optional[Any] = None,
        use_amp: bool = False,
        use_ddp: bool = False,
        ddp_backend: Optional[str] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        optimizer_name: str = "adamw",
        scheduler_name: str = "none",
        warmup_steps: int = 0,
        total_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        max_eval_step: Optional[int] = None,
        training_config: Optional[TrainingConfig] = None,
        sharded_checkpoint: bool = False,
        broadcast_weights_from_rank0: bool = False,
    ) -> None:
        if model is None:
            raise ValueError("`model` must be provided.")

        self.device = get_device(device) if device == "auto" else torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.scheduler = scheduler
        self.criterion = criterion or language_modeling_cross_entropy

        self.seed = seed
        cfg = training_config or TrainingConfig(
            max_epochs=max_epochs,
            max_train_steps=max_train_steps if max_train_steps is not None else max_steps,
            max_eval_steps=max_eval_steps if max_eval_steps is not None else max_eval_step,
            eval_every_n_epochs=eval_every_n_epochs,
            save_every_n_epochs=save_every_n_epochs,
            grad_accumulation_step=grad_accumulation_step,
            max_grad_norm=max_grad_norm,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        self.max_epochs = int(cfg.max_epochs)
        self.max_train_steps = cfg.max_train_steps
        self.max_eval_steps = cfg.max_eval_steps
        self.eval_every_n_epochs = max(1, int(cfg.eval_every_n_epochs))
        self.save_every_n_epochs = max(1, int(cfg.save_every_n_epochs))
        self.grad_accumulation_step = max(1, int(cfg.grad_accumulation_step))
        self.max_grad_norm = cfg.max_grad_norm
        self.monitor = monitor
        self.use_amp = bool(use_amp)
        self.scaler = scaler
        self.use_ddp = bool(use_ddp)
        self.ddp_backend = ddp_backend
        self.resume_from = resume_from
        self.checkpoint_dir = checkpoint_dir
        self.sharded_checkpoint = bool(sharded_checkpoint)
        self.broadcast_weights_from_rank0 = bool(broadcast_weights_from_rank0)

        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.optimizer_name = cfg.optimizer_name
        self.scheduler_name = cfg.scheduler_name
        self.warmup_steps = cfg.warmup_steps
        self.total_steps = cfg.total_steps

        self._setup_seed()
        self._setup_distributed()
        self._setup_device()
        self._setup_scaler()
        self._setup_optimizer()
        self._validate_optimizer_device()
        self._setup_scheduler()
        self._setup_monitor()
        self._setup_checkpoint_manager()

        self.state = TrainingState(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
            config={"criterion": self.criterion},
        )

        self.engine = Engine(
            state=self.state,
            grad_accum_steps=self.grad_accumulation_step,
            max_grad_norm=self.max_grad_norm,
            monitor=self.monitor,
            max_train_steps=self.max_train_steps,
        )
        self.engine.max_eval_steps = self.max_eval_steps

        if self.resume_from:
            self.load(self.resume_from)

    def _setup_seed(self) -> None:
        if self.seed is not None:
            seed_everything(self.seed)

    def _setup_distributed(self) -> None:
        if not self.use_ddp:
            return

        try:
            init_distributed(backend=self.ddp_backend)
        except Exception as exc:
            print_once(f"[Trainer] Distributed initialization skipped: {exc}")

    def _setup_device(self) -> None:
        if self.use_ddp and is_distributed():
            self.model = wrap_model_ddp(self.model)
        print_device_info()

    def _validate_optimizer_device(self) -> None:
        if self.optimizer is None:
            return

        model_device = next(self.model.parameters()).device

        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.device != model_device:
                    raise ValueError(
                        "The supplied optimizer references parameters on "
                        f"{param.device}, but the model is on {model_device}.\n\n"
                        "Move the model before creating the optimizer, or let Trainer create the optimizer."
                    )

        for state_dict in self.optimizer.state.values():
            for k, v in state_dict.items():
                if torch.is_tensor(v) and v.device != model_device:
                    state_dict[k] = v.to(model_device)

    def _setup_scaler(self) -> None:
        if self.scaler is not None:
            return
        self.scaler = initialize_scaler(enabled=self.use_amp)

    def _setup_monitor(self) -> None:
        if self.monitor is not None:
            return
        if not is_main_process():
            self.monitor = None
            return
        self.monitor = Logger(
            csv=True, tensorboard=False, wandb=False, log_dir="logs", experiment_name="stackformer"
        )

    def _setup_optimizer(self) -> None:
        if self.optimizer is not None:
            return
        self.optimizer = create_optimizer(
            self.model,
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _setup_scheduler(self) -> None:
        if self.scheduler is not None:
            return
        if self.scheduler_name is None or self.scheduler_name == "none":
            self.scheduler = None
            return

        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_name=self.scheduler_name,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
        )

    def _setup_checkpoint_manager(self) -> None:
        self.checkpoint_manager = CheckpointManager(output_dir=self.checkpoint_dir, device=self.device)

    def _build_resume_dataloader(
        self,
        dataloader: Optional[DataLoader],
    ) -> Optional[DataLoader]:
        """Rebuild a dataloader for deterministic checkpoint resumption.

        Args:
            dataloader (DataLoader | None): Base DataLoader instance.

        Returns:
            DataLoader | None: Resumed DataLoader positioned at the correct sample index.
        """
        if dataloader is None or self.state.batch_idx == 0:
            return dataloader

        dataset = dataloader.dataset
        batch_size = dataloader.batch_size

        if batch_size is None:
            return dataloader

        start_sample = self.state.batch_idx * batch_size

        if start_sample >= len(dataset):
            return dataloader

        sampler = getattr(dataloader, "sampler", None)

        if (
            self.seed is not None
            and isinstance(sampler, torch.utils.data.RandomSampler)
            and sampler.replacement is False
        ):
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.state.epoch)

            permutation = torch.randperm(
                len(dataset),
                generator=generator,
            ).tolist()

            remaining_indices = permutation[start_sample:]

            resumed_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(remaining_indices),
                num_workers=dataloader.num_workers,
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                collate_fn=dataloader.collate_fn,
                persistent_workers=getattr(
                    dataloader,
                    "persistent_workers",
                    False,
                ),
            )

            print_once(
                f"[Trainer] Resuming shuffled dataloader from sample {start_sample}/{len(dataset)}"
            )

            return resumed_loader

        subset = Subset(dataset, range(start_sample, len(dataset)))

        resumed_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            collate_fn=dataloader.collate_fn,
            persistent_workers=getattr(
                dataloader,
                "persistent_workers",
                False,
            ),
        )

        print_once(
            "[Trainer] Warning: exact shuffle resumption is not available for this DataLoader. Falling back to sequential resume."
        )
        return resumed_loader

    def fit(self) -> None:
        """Run full model training loop across configured epochs."""
        if self.train_loader is None:
            raise ValueError("train_dataloader is required to call fit().")

        if self.state.epoch >= self.max_epochs:
            print_once(
                f"[Trainer] Warning: Resumed at epoch {self.state.epoch}, but max_epochs is {self.max_epochs}. "
                "Training skipped. Increase max_epochs to continue training."
            )
            return

        for epoch in range(self.state.epoch, self.max_epochs):
            train_loader = (
                self._build_resume_dataloader(self.train_loader)
                if epoch == self.state.epoch and self.state.batch_idx > 0
                else self.train_loader
            )
            self.engine.train_one_epoch(train_loader, epoch)

            if self.engine.reached_max_train_steps():
                self.save("latest")
                break

            if self.val_loader is not None and ((epoch + 1) % self.eval_every_n_epochs == 0):
                self.engine.validate_one_epoch(self.val_loader, epoch)

            self.state.reset_batch()
            self.state.next_epoch()

            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.save("latest")

    def validate(self) -> None:
        """Run validation loop on the validation dataset."""
        if self.val_loader is None:
            raise ValueError("Validation dataloader not provided.")
        self.engine.validate_one_epoch(self.val_loader, self.state.epoch)

    def save(self, name: str = "latest", export_jit: bool = False) -> None:
        """Save checkpoint to disk.

        Args:
            name (str, default="latest"): Checkpoint name identifier.
            export_jit (bool, default=False): Export TorchScript model alongside checkpoint.
        """
        jit_path = None

        if export_jit:
            try:
                jit_path = self.checkpoint_manager.save_jit_model(
                    self.model,
                    f"{name}_model",
                )
            except Exception as exc:
                print_once(f"[Trainer] TorchScript export skipped: {exc}")

        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "batch_idx": self.state.batch_idx,
            "seed": self.seed,
            "config": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "optimizer_name": self.optimizer_name,
                "scheduler_name": self.scheduler_name,
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps,
                "criterion": getattr(self.criterion, "__name__", str(self.criterion)),
            },
            "jit_model_path": jit_path,
        }

        if self.sharded_checkpoint:
            self.checkpoint_manager.save_sharded(state, name)
        else:
            self.checkpoint_manager.save(state, name)

    def load(self, checkpoint: str = "latest") -> None:
        """Restore trainer model, optimizer, scheduler, and step metadata from checkpoint.

        Args:
            checkpoint (str, default="latest"): Checkpoint name identifier.
        """
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scaler": self.scaler,
        }

        if self.sharded_checkpoint:
            metadata = self.checkpoint_manager.load_sharded(checkpoint, state)
        else:
            metadata = self.checkpoint_manager.load(
                checkpoint,
                state,
                broadcast_from_rank0=self.broadcast_weights_from_rank0,
            )

        self.state.load_metadata(metadata)

        if metadata.get("seed") is not None:
            self.seed = metadata["seed"]

        print_once(f"[Trainer] Resumed from checkpoint '{checkpoint}'")

    def export_torchscript(self, name: str = "inference_model") -> str:
        """Export current model as TorchScript artifact and return path."""
        return self.checkpoint_manager.save_jit_model(self.model, name=name)