"""Core Accelerate-style Trainer for Stackformer."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager, clean_state_dict, unwrap_model
from .config import TrainConfig
from .evaluation import run_evaluation
from .events import (
    close_loggers,
    dispatch_checkpoint_save,
    dispatch_epoch_end,
    dispatch_epoch_start,
    dispatch_step_end,
    dispatch_train_end,
    dispatch_train_start,
)
from .losses import get_loss
from .optim import build_optimizer, build_scheduler
from .parallelize import apply_model_transformations
from .precision import PrecisionManager
from .progress import ProgressBar
from .state import TrainState
from .utils import forward_batch, prepare_inputs
from stackformer.utils.checkpoint import save_model_weights


class Trainer:
    """Small, readable, Accelerate-like Trainer for Stackformer models."""

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

        if config is None:
            self.config = TrainConfig(**kwargs) if kwargs else TrainConfig()
        elif kwargs:
            cfg_dict = config.to_dict()
            cfg_dict.update(kwargs)
            self.config = TrainConfig.from_dict(cfg_dict)
        else:
            self.config = copy.deepcopy(config)

        if loss is not None:
            self.config.loss = loss

        self.device = self.config.resolve_device()
        self.model.to(self.device)

        self.model = apply_model_transformations(self.model, self.config)

        if isinstance(self.config.loss, str):
            self.loss_fn = get_loss(self.config.loss, loss_mode=self.config.loss_mode)
        else:
            self.loss_fn = self.config.loss

        self.precision = PrecisionManager(self.config.mixed_precision, self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer or build_optimizer(self.model, self.config)
        total_steps = self._estimate_total_steps()
        self.scheduler = scheduler or build_scheduler(self.optimizer, self.config, total_steps)

        self.state = TrainState()
        self.metrics, self.loggers, self.callbacks = list(metrics or []), list(loggers or []), list(callbacks or [])

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir, keep_last_n=self.config.keep_last_n,
            save_best=self.config.save_best, monitor=self.config.monitor, mode=self.config.mode,
            save_best_state=self.config.save_best_state, async_checkpoint=self.config.async_checkpoint, use_dcp=(self.config.parallel == "fsdp2"),
        )

    def _estimate_total_steps(self) -> int:
        if self.config.max_steps is not None:
            return self.config.max_steps
        if self.train_loader is not None and hasattr(self.train_loader, "__len__"):
            steps_per_epoch = max(1, len(self.train_loader) // max(1, self.config.grad_accum_steps))
            return max(1, (self.config.max_epochs or 1) * steps_per_epoch)
        return 1000

    def _prepare_inputs(self, batch: Any) -> Any:
        return prepare_inputs(batch, self.device)

    def _forward(self, batch: Any) -> torch.Tensor:
        return forward_batch(self.model, batch)

    def _restore_checkpoint_if_requested(self) -> bool:
        if not self.config.resume_from:
            return False
        ckpt = self.checkpoint_manager.load(self.config.resume_from, model=self.model, optimizer=self.optimizer)
        if ckpt is None:
            return False
        unwrap_model(self.model).load_state_dict(clean_state_dict(ckpt["model"]))
        for key, target in [("optimizer", self.optimizer), ("scheduler", self.scheduler)]:
            if target is not None and ckpt.get(key) is not None:
                target.load_state_dict(ckpt[key])
        if ckpt.get("scaler") is not None and hasattr(self.precision.scaler, "load_state_dict"):
            self.precision.scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("rng"):
            self.checkpoint_manager._set_rng_state(ckpt["rng"])
        if ckpt.get("train_state"):
            self.state.load_dict(ckpt["train_state"])
        return True

    def train(self) -> TrainState:
        """Execute the training loop with support for mid-epoch resume, callbacks, and metrics."""
        if self.train_loader is None:
            raise ValueError("Cannot call trainer.train() without a train_loader.")

        resumed = self._restore_checkpoint_if_requested()

        if getattr(self.train_loader, "shuffle", False) or (
            hasattr(self.train_loader, "sampler")
            and isinstance(getattr(self.train_loader, "sampler", None), torch.utils.data.RandomSampler)
        ):
            if getattr(self.train_loader, "generator", None) is None:
                warnings.warn(
                    "DataLoader has shuffle=True but no generator configured with a fixed seed. "
                    "For bit-identical mid-epoch resume, instantiate DataLoader with a seeded torch.Generator.",
                    UserWarning,
                    stacklevel=2,
                )

        self.model.train()
        self.state.should_stop = False
        accum_steps = self.config.grad_accum_steps
        max_steps = self.config.max_steps
        max_epochs = self.config.max_epochs

        dispatch_train_start(self)

        try:
            epoch = self.state.epoch
            while True:
                if (max_epochs is not None and epoch >= max_epochs) or (
                    max_steps is not None and self.state.global_step >= max_steps
                ):
                    break

                self.state.epoch = epoch
                skip_batches = self.state.step_in_epoch if resumed else 0
                resumed = False

                dispatch_epoch_start(self, epoch)

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

                    with self.precision.autocast():
                        outputs = self._forward(batch)
                        loss = self.loss_fn(outputs, batch)
                        scaled_loss = loss / accum_steps

                    self.precision.scale_loss(scaled_loss).backward()
                    accum_loss += loss.item()

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
                        dispatch_step_end(self, self.state.global_step, {"loss": running_loss, "lr": lr_curr})

                        if self.config.save_every_n_steps and self.state.global_step % self.config.save_every_n_steps == 0:
                            self.save_checkpoint()
                        if self.config.eval_every_n_steps and self.state.global_step % self.config.eval_every_n_steps == 0:
                            self.evaluate()
                            self.model.train()

                        if self.state.should_stop or (max_steps is not None and self.state.global_step >= max_steps):
                            self.state.should_stop = True
                            break

                pbar.close()

                dispatch_epoch_end(self, epoch, {"epoch_loss": running_loss})

                if self.config.save_every_n_epochs and (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self.save_checkpoint()
                if self.config.eval_every_n_epochs and (epoch + 1) % self.config.eval_every_n_epochs == 0:
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
            dispatch_train_end(self)
            self.close()

        return self.state

    def evaluate(self) -> dict[str, float]:
        """Run evaluation over the validation loader and compute all registered metrics."""
        return run_evaluation(self)

    def save_checkpoint(self) -> Path:
        """Save a full training checkpoint to the checkpoint directory."""
        path = self.checkpoint_manager.save(
            model=self.model, train_state=self.state, optimizer=self.optimizer,
            scheduler=self.scheduler, scaler=self.precision.scaler, config=self.config,
        )
        dispatch_checkpoint_save(self, str(path))
        return path

    def save_best_model(self, metrics: dict[str, float] | None = None) -> Path:
        """Save the best-performing model weights to the best/ directory."""
        return self.checkpoint_manager.save_best(
            model=self.model, train_state=self.state, metrics=metrics,
            optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.precision.scaler,
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
        close_loggers(self)
