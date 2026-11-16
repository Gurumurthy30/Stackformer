"""Core Accelerate-style Trainer for Stackformer."""

from __future__ import annotations

import copy
import random
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stackformer.training import events
from stackformer.utils.checkpoint import save_model_weights

from .checkpoint import CheckpointManager
from .config import TrainConfig
from .display import DisplayManager
from .evaluation import run_evaluation
from .losses import get_loss
from .optim import build_optimizer, build_scheduler
from .parallelize import apply_model_transformations
from .precision import PrecisionManager
from .state import TrainState
from .utils import forward_batch, prepare_inputs, warn_if_unseeded_shuffle


def _seed_training_rngs(seed: int) -> None:
    """Seed random, numpy, and torch RNGs for training reproducibility.

    Note:
        Seeding in Trainer cannot affect model weight initialization performed
        earlier by the caller.
    """
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Small, readable, Accelerate-like Trainer for Stackformer models.

    Note:
        Random seeds in `TrainConfig.seed` are applied during `Trainer` initialization
        to seed Python, NumPy, and PyTorch RNGs for training reproducibility (dropout,
        data shuffling, sampling). This cannot affect model weight initialization
        performed earlier by the caller.
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

        if kwargs:
            valid_fields = {f.name for f in TrainConfig.__dataclass_fields__.values()}
            unknown = set(kwargs) - valid_fields
            if unknown:
                raise TypeError(f"Unknown Trainer/TrainConfig argument(s): {sorted(unknown)}")

        if config is None:
            self.config = TrainConfig(**kwargs) if kwargs else TrainConfig()
        else:
            self.config = (
                TrainConfig.from_dict({**config.to_dict(), **kwargs})
                if kwargs
                else copy.deepcopy(config)
            )

        _seed_training_rngs(self.config.seed)

        self._closed = False
        if self.config.metrics_on_train:
            warnings.warn(
                "TrainConfig.metrics_on_train is not implemented and will be ignored. "
                "Training metrics are not computed per batch.",
                UserWarning,
                stacklevel=2,
            )

        if loss is not None:
            self.config.loss = loss

        self.device = self.config.resolve_device()
        self.model.to(self.device)
        self.model = apply_model_transformations(self.model, self.config)

        self.loss_fn = (
            get_loss(self.config.loss, loss_mode=self.config.loss_mode)
            if isinstance(self.config.loss, str)
            else self.config.loss
        )
        self.precision = PrecisionManager(self.config.mixed_precision, self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer or build_optimizer(self.model, self.config)
        total_steps = self._estimate_total_steps()
        self.scheduler = scheduler or build_scheduler(self.optimizer, self.config, total_steps)

        self.state = TrainState()
        self.metrics, self.loggers, self.callbacks = (
            list(metrics or []),
            list(loggers or []),
            list(callbacks or []),
        )
        self.display = DisplayManager(
            mode=self.config.display, verbosity=self.config.verbosity, rank_zero_only=True
        )

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            keep_last_n=self.config.keep_last_n,
            save_best=self.config.save_best,
            monitor=self.config.monitor,
            mode=self.config.mode,
            save_best_state=self.config.save_best_state,
            async_checkpoint=self.config.async_checkpoint,
            use_dcp=(self.config.parallel == "fsdp2"),
        )

    def _estimate_total_steps(self) -> int:
        if self.config.max_steps is not None:
            return self.config.max_steps
        if self.train_loader is not None and hasattr(self.train_loader, "__len__"):
            return max(
                1,
                (self.config.max_epochs or 1)
                * max(1, len(self.train_loader) // max(1, self.config.grad_accum_steps)),
            )
        return 1000

    def _prepare_inputs(self, batch: Any) -> Any:
        return prepare_inputs(batch, self.device)

    def _forward(self, batch: Any) -> torch.Tensor:
        return forward_batch(self.model, batch)

    def _optimizer_step(self, num_accumulated: int, accum_loss: float) -> float:
        accum_steps = self.config.grad_accum_steps
        self.precision.unscale_(self.optimizer)

        if num_accumulated != accum_steps:
            scale_factor = accum_steps / num_accumulated
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale_factor)

        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.precision.step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()

        self.state.global_step += 1
        step_loss = accum_loss / num_accumulated

        lr_curr = self.optimizer.param_groups[0]["lr"]
        events.dispatch_step_end(self, self.state.global_step, {"loss": step_loss, "lr": lr_curr})

        step = self.state.global_step
        if self.config.save_every_n_steps and step % self.config.save_every_n_steps == 0:
            self.save_checkpoint()
        if self.config.eval_every_n_steps and step % self.config.eval_every_n_steps == 0:
            self.evaluate()
            self.model.train()

        return step_loss

    def _restore_checkpoint_if_requested(self) -> bool:
        return bool(
            self.config.resume_from
            and self.checkpoint_manager.restore_training_state(
                self.config.resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.precision.scaler,
                self.state,
            )
        )

    def train(self) -> TrainState:
        """Execute the training loop with support for mid-epoch resume, callbacks, and metrics."""
        if self.train_loader is None:
            raise ValueError("Cannot call trainer.train() without a train_loader.")

        resumed = self._restore_checkpoint_if_requested()
        warn_if_unseeded_shuffle(self.train_loader)

        self.model.train()
        self.state.should_stop = False
        accum_steps = self.config.grad_accum_steps
        max_steps = self.config.max_steps
        max_epochs = self.config.max_epochs

        events.dispatch_train_start(self)

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

                events.dispatch_epoch_start(self, epoch)

                sampler = getattr(self.train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    gen = getattr(self.train_loader, "generator", None)
                    if gen is not None:
                        gen.manual_seed(self.config.seed + epoch)

                pbar = self.display.progress_bar(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}" + (f"/{max_epochs}" if max_epochs else ""),
                )

                running_loss = 0.0
                accum_loss = 0.0
                accum_count = 0
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
                    accum_count += 1

                    if accum_count == accum_steps:
                        running_loss = self._optimizer_step(
                            num_accumulated=accum_steps, accum_loss=accum_loss
                        )
                        accum_loss = 0.0
                        accum_count = 0

                        lr_curr = self.optimizer.param_groups[0]["lr"]
                        pbar.set_postfix(loss=running_loss, lr=lr_curr, step=self.state.global_step)

                        if self.state.should_stop or (
                            max_steps is not None and self.state.global_step >= max_steps
                        ):
                            self.state.should_stop = True
                            break

                pbar.close()

                # Flush any trailing micro-batches at epoch end (#35)
                if accum_count > 0:
                    running_loss = self._optimizer_step(
                        num_accumulated=accum_count, accum_loss=accum_loss
                    )
                    accum_loss = 0.0
                    accum_count = 0

                events.dispatch_epoch_end(self, epoch, {"epoch_loss": running_loss})

                ep_1 = epoch + 1
                if self.config.save_every_n_epochs and ep_1 % self.config.save_every_n_epochs == 0:
                    self.save_checkpoint()
                if self.config.eval_every_n_epochs and ep_1 % self.config.eval_every_n_epochs == 0:
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
            events.dispatch_train_end(self)
            events.flush_loggers(self)

        return self.state

    def evaluate(self) -> dict[str, float]:
        """Run evaluation over the validation loader and compute all registered metrics."""
        return run_evaluation(self)

    def save_checkpoint(self) -> Path:
        """Save a full training checkpoint to the checkpoint directory."""
        path = self.checkpoint_manager.save(
            self.model,
            self.state,
            self.optimizer,
            self.scheduler,
            self.precision.scaler,
            self.config,
        )
        events.dispatch_checkpoint_save(self, str(path))
        return path

    def save_best_model(self, metrics: dict[str, float] | None = None) -> Path:
        """Save the best-performing model weights to the best/ directory."""
        return self.checkpoint_manager.save_best(
            self.model,
            self.state,
            metrics,
            self.optimizer,
            self.scheduler,
            self.precision.scaler,
            self.config,
        )

    def load_best_model(self) -> None:
        """Load best-performing model weights into the active model."""
        self.checkpoint_manager.load_best(self.model, device=self.device)

    def save(self, path: str | Path) -> None:
        """Export standalone model weights compatible with BaseModel.load_weights."""
        save_model_weights(self.model, path)

    def close(self) -> None:
        """Close and flush all active loggers (idempotent)."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        events.close_loggers(self)
