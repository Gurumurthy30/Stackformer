"""Core Accelerate-style Trainer for Stackformer."""

from __future__ import annotations

import copy
from typing import Any, Callable, Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .losses import get_loss
from .optim import build_optimizer, build_scheduler
from .precision import PrecisionManager
from .progress import ProgressBar
from .state import TrainState


class Trainer:
    """Simple, readable, Accelerate-like Trainer for Stackformer models.

    Args:
        model: nn.Module to train.
        config: TrainConfig hyperparameter dataclass.
        train_loader: Training DataLoader or iterable.
        val_loader: Validation DataLoader or iterable.
        optimizer: Custom optimizer (overrides config-built optimizer).
        scheduler: Custom scheduler (overrides config-built scheduler).
        loss: Optional loss function name or callable override.
        metrics: Optional list of metrics (v1 hook).
        loggers: Optional list of loggers (v1 hook).
        callbacks: Optional list of callbacks (v1 hook).
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

        # Training state & hooks
        self.state = TrainState()
        self.metrics = list(metrics or [])
        self.loggers = list(loggers or [])
        self.callbacks = list(callbacks or [])

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
        """Execute the training loop."""
        if self.train_loader is None:
            raise ValueError("Cannot call trainer.train() without a train_loader.")

        self.model.train()
        self.state.should_stop = False

        accum_steps = self.config.grad_accum_steps
        max_steps = self.config.max_steps
        max_epochs = self.config.max_epochs

        # Epoch loop
        epoch = self.state.epoch
        while True:
            if max_epochs is not None and epoch >= max_epochs:
                break
            if max_steps is not None and self.state.global_step >= max_steps:
                break

            self.state.epoch = epoch
            self.state.step_in_epoch = 0

            # Set distributed sampler epoch if applicable
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            pbar = ProgressBar(
                self.train_loader,
                desc=f"Epoch {epoch + 1}" + (f"/{max_epochs}" if max_epochs else ""),
                enabled=True,
            )

            running_loss = 0.0
            accum_loss = 0.0

            for batch in pbar:
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
                    # Clip gradients
                    if self.config.max_grad_norm is not None:
                        self.precision.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    # Optimizer & Scheduler step
                    self.precision.step(self.optimizer)
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.state.global_step += 1
                    running_loss = accum_loss / accum_steps
                    accum_loss = 0.0

                    lr_curr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(loss=running_loss, lr=lr_curr, step=self.state.global_step)

                    # Step-based evaluation
                    if (
                        self.config.eval_every_n_steps is not None
                        and self.state.global_step % self.config.eval_every_n_steps == 0
                    ):
                        self.evaluate()
                        self.model.train()

                    if max_steps is not None and self.state.global_step >= max_steps:
                        self.state.should_stop = True
                        break

            pbar.close()

            # Epoch-based evaluation
            if (
                self.config.eval_every_n_epochs is not None
                and (epoch + 1) % self.config.eval_every_n_epochs == 0
            ):
                self.evaluate()
                self.model.train()

            epoch += 1
            if self.state.should_stop:
                break

        return self.state

    def evaluate(self) -> dict[str, float]:
        """Run evaluation over the validation loader."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        max_eval = self.config.max_eval_steps

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

        avg_loss = total_loss / max(1, num_batches)
        metrics = {"val_loss": avg_loss}

        if self.config.save_best and self.config.monitor == "val_loss":
            self.state.update_best(avg_loss, mode=self.config.mode)

        return metrics

