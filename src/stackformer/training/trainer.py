"""Core Accelerate-style Trainer for Stackformer."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager
from .config import TrainConfig
from .losses import get_loss
from .optim import build_optimizer, build_scheduler
from .precision import PrecisionManager
from .progress import ProgressBar
from .state import TrainState
from stackformer.utils.checkpoint import save_model_weights


class Trainer:
    """Small, readable, Accelerate-like Trainer for Stackformer models.

    Args:
        model: nn.Module to train.
        config: TrainConfig hyperparameter dataclass.
        train_loader: Training DataLoader or iterable.
        val_loader: Validation DataLoader or iterable.
        optimizer: Custom optimizer (overrides config-built optimizer).
        scheduler: Custom scheduler (overrides config-built scheduler).
        loss: Optional loss function name or callable override.
        metrics: Optional sequence of evaluation metrics.
        loggers: Optional sequence of training loggers.
        callbacks: Optional sequence of lifecycle callbacks.
        **kwargs: Loose hyperparameter overrides for TrainConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig | None = None,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        loss: str | Callable[[Any, Any], torch.Tensor] | None = None,
        metrics: Sequence[Any] | None = None,
        loggers: Sequence[Any] | None = None,
        callbacks: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model

        # Merge config and loose kwargs
        if config is None:
            self.config = TrainConfig(**kwargs) if kwargs else TrainConfig()
        else:
            if kwargs:
                cfg_dict = config.to_dict()
                cfg_dict.update(kwargs)
                self.config = TrainConfig.from_dict(cfg_dict)
            else:
                self.config = copy.deepcopy(config)

        if loss is not None:
            self.config.loss = loss

        # Device & Model
        self.device = self.config.resolve_device()
        self.model.to(self.device)

        # Loss function
        if isinstance(self.config.loss, str):
            self.loss_fn = get_loss(self.config.loss, loss_mode=self.config.loss_mode)
        else:
            self.loss_fn = self.config.loss

        # Precision
        self.precision = PrecisionManager(self.config.mixed_precision, self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = optimizer or build_optimizer(self.model, self.config)

        # Scheduler steps calculation
        total_steps = self._estimate_total_steps()
        self.scheduler = scheduler or build_scheduler(self.optimizer, self.config, total_steps)

        # State, checkpoints, metrics, loggers, callbacks
        self.state = TrainState()
        self.metrics = list(metrics or [])
        self.loggers = list(loggers or [])
        self.callbacks = list(callbacks or [])

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            keep_last_n=self.config.keep_last_n,
            save_best=self.config.save_best,
            monitor=self.config.monitor,
            mode=self.config.mode,
            save_best_state=self.config.save_best_state,
            async_checkpoint=self.config.async_checkpoint,
        )

    def _estimate_total_steps(self) -> int:
        if self.config.max_steps is not None:
            return self.config.max_steps
        if self.train_loader is not None and hasattr(self.train_loader, "__len__"):
            steps_per_epoch = max(1, len(self.train_loader) // max(1, self.config.grad_accum_steps))
            return max(1, (self.config.max_epochs or 1) * steps_per_epoch)
        return 1000

    def _prepare_inputs(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (tuple, list)):
            return tuple(self._prepare_inputs(x) for x in batch)
        elif isinstance(batch, dict):
            return {k: self._prepare_inputs(v) for k, v in batch.items()}
        return batch

    def _forward(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return self.model(batch[0])
        elif isinstance(batch, dict):
            if "input_ids" in batch:
                return self.model(batch["input_ids"])
            elif "pixel_values" in batch:
                return self.model(batch["pixel_values"])
            return self.model(**batch)
        return self.model(batch)

    def train(self) -> TrainState:
        """Execute the training loop with support for mid-epoch resume, callbacks, and metrics."""
        if self.train_loader is None:
            raise ValueError("Cannot call trainer.train() without a train_loader.")

        # Resume from checkpoint if requested
        resumed = False
        if self.config.resume_from:
            ckpt = self.checkpoint_manager.load(self.config.resume_from)
            if ckpt is not None:
                self.model.load_state_dict(ckpt["model"])
                if self.optimizer is not None and ckpt.get("optimizer") is not None:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                if self.scheduler is not None and ckpt.get("scheduler") is not None:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
                if ckpt.get("scaler") is not None and hasattr(self.precision.scaler, "load_state_dict"):
                    self.precision.scaler.load_state_dict(ckpt["scaler"])
                if ckpt.get("rng"):
                    self.checkpoint_manager._set_rng_state(ckpt["rng"])
                if ckpt.get("train_state"):
                    self.state.load_dict(ckpt["train_state"])
                resumed = True

        # Check for unseeded shuffled DataLoader
        if getattr(self.train_loader, "shuffle", False) or (
            hasattr(self.train_loader, "sampler")
            and isinstance(getattr(self.train_loader, "sampler", None), torch.utils.data.RandomSampler)
        ):
            gen = getattr(self.train_loader, "generator", None)
            if gen is None:
                warnings.warn(
                    "DataLoader has shuffle=True but no generator configured with a fixed seed. "
                    "For bit-identical mid-epoch resume, instantiate DataLoader with a seeded torch.Generator "
                    "(e.g., DataLoader(..., generator=torch.Generator().manual_seed(seed))).",
                    UserWarning,
                    stacklevel=2,
                )

        self.model.train()
        self.state.should_stop = False

        accum_steps = self.config.grad_accum_steps
        max_steps = self.config.max_steps
        max_epochs = self.config.max_epochs

        # Callbacks & Loggers train start
        for cb in self.callbacks:
            cb.on_train_start(self)

        cfg_dict = self.config.to_dict()
        cfg_dict["num_parameters"] = sum(p.numel() for p in self.model.parameters())
        for lg in self.loggers:
            lg.log_config(cfg_dict)

        try:
            epoch = self.state.epoch
            while True:
                if max_epochs is not None and epoch >= max_epochs:
                    break
                if max_steps is not None and self.state.global_step >= max_steps:
                    break

                self.state.epoch = epoch
                skip_batches = self.state.step_in_epoch if resumed else 0
                resumed = False  # Only skip on the first resumed epoch

                for cb in self.callbacks:
                    cb.on_epoch_start(self, epoch)

                # Sampler / generator epoch determinism
                if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(epoch)
                elif getattr(self.train_loader, "generator", None) is not None:
                    self.train_loader.generator.manual_seed(self.config.seed + epoch)

                pbar = ProgressBar(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}" + (f"/{max_epochs}" if max_epochs else ""),
                    enabled=True,
                )

                running_loss = 0.0
                accum_loss = 0.0
                skipped = 0

                for batch in pbar:
                    if skipped < skip_batches:
                        skipped += 1
                        continue

                    batch = self._prepare_inputs(batch)
                    self.state.total_micro_batches += 1
                    self.state.step_in_epoch += 1

                    # Forward with AMP
                    with self.precision.autocast():
                        outputs = self._forward(batch)
                        loss = self.loss_fn(outputs, batch)
                        scaled_loss = loss / accum_steps

                    # Backward
                    self.precision.scale_loss(scaled_loss).backward()
                    accum_loss += loss.item()

                    # Step optimizer on accumulation boundary
                    if self.state.total_micro_batches % accum_steps == 0:
                        if self.config.max_grad_norm is not None:
                            self.precision.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.max_grad_norm
                            )

                        self.precision.step(self.optimizer)
                        self.optimizer.zero_grad(set_to_none=True)

                        if self.scheduler is not None:
                            self.scheduler.step()

                        self.state.global_step += 1
                        running_loss = accum_loss / accum_steps
                        accum_loss = 0.0

                        lr_curr = self.optimizer.param_groups[0]["lr"]
                        pbar.set_postfix(loss=running_loss, lr=lr_curr, step=self.state.global_step)

                        step_metrics = {"loss": running_loss, "lr": lr_curr}
                        for cb in self.callbacks:
                            cb.on_step_end(self, self.state.global_step, step_metrics)
                        for lg in self.loggers:
                            lg.log_metrics(step_metrics, step=self.state.global_step)

                        # Step-based saving & evaluation
                        if (
                            self.config.save_every_n_steps is not None
                            and self.state.global_step % self.config.save_every_n_steps == 0
                        ):
                            self.save_checkpoint()

                        if (
                            self.config.eval_every_n_steps is not None
                            and self.state.global_step % self.config.eval_every_n_steps == 0
                        ):
                            self.evaluate()
                            self.model.train()

                        if self.state.should_stop or (max_steps is not None and self.state.global_step >= max_steps):
                            self.state.should_stop = True
                            break

                pbar.close()

                epoch_metrics = {"epoch_loss": running_loss}
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, epoch_metrics)

                # Epoch-based saving & evaluation
                if (
                    self.config.save_every_n_epochs is not None
                    and (epoch + 1) % self.config.save_every_n_epochs == 0
                ):
                    self.save_checkpoint()

                if (
                    self.config.eval_every_n_epochs is not None
                    and (epoch + 1) % self.config.eval_every_n_epochs == 0
                ):
                    self.evaluate()
                    self.model.train()

                self.state.step_in_epoch = 0
                epoch += 1
                if self.state.should_stop:
                    break

        except KeyboardInterrupt:
            if self.config.save_last:
                self.save_checkpoint()
            raise
        finally:
            for cb in self.callbacks:
                try:
                    cb.on_train_end(self)
                except Exception:
                    pass
            self.close()

        return self.state

    def evaluate(self) -> dict[str, float]:
        """Run evaluation over the validation loader and compute all registered metrics."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        max_eval = self.config.max_eval_steps

        for m in self.metrics:
            m.reset()

        with torch.no_grad():
            for batch in self.val_loader:
                if max_eval is not None and num_batches >= max_eval:
                    break
                batch = self._prepare_inputs(batch)
                with self.precision.autocast():
                    outputs = self._forward(batch)
                    loss = self.loss_fn(outputs, batch)

                total_loss += loss.item()
                num_batches += 1

                for m in self.metrics:
                    m.update(outputs, batch)

        avg_loss = total_loss / max(1, num_batches)
        eval_metrics = {"val_loss": avg_loss}

        for m in self.metrics:
            res = m.compute()
            if isinstance(res, dict):
                for k, v in res.items():
                    eval_metrics[f"val_{k}"] = v
            else:
                eval_metrics[f"val_{m.name}"] = res

        # Callbacks & Loggers
        for cb in self.callbacks:
            cb.on_eval_end(self, eval_metrics)
        for lg in self.loggers:
            lg.log_metrics(eval_metrics, step=self.state.global_step)

        # Best-model checkpointing
        if self.config.save_best and self.config.monitor in eval_metrics:
            mon_val = eval_metrics[self.config.monitor]
            if self.state.update_best(mon_val, mode=self.config.mode):
                self.save_best_model(eval_metrics)

        return eval_metrics

    def save_checkpoint(self) -> Path:
        """Save a full training checkpoint to the checkpoint directory."""
        path = self.checkpoint_manager.save(
            model=self.model,
            train_state=self.state,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.precision.scaler,
            config=self.config,
        )
        for cb in self.callbacks:
            cb.on_checkpoint_save(self, str(path))
        return path

    def save_best_model(self, metrics: dict[str, float] | None = None) -> Path:
        """Save the best-performing model weights to the best/ directory."""
        return self.checkpoint_manager.save_best(
            model=self.model,
            train_state=self.state,
            metrics=metrics,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.precision.scaler,
            config=self.config,
        )

    def load_best_model(self) -> None:
        """Load best-performing model weights into the active model."""
        self.checkpoint_manager.load_best(self.model, device=self.device)

    def save(self, path: str | Path) -> None:
        """Export standalone model weights compatible with BaseModel.load_weights."""
        save_model_weights(self.model, path)

    def close(self) -> None:
        """Close and flush all active loggers."""
        for lg in self.loggers:
            try:
                lg.finish()
            except Exception:
                pass
