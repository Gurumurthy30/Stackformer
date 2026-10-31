"""Runtime step execution engine for training and evaluation.

Executes forward/backward optimization steps, gradient accumulation, AMP scaling, and metric tracking.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

from stackformer.amp import step_optimizer, update_scaler
from stackformer.logging.metrics import MetricTracker
from stackformer.training.loops import eval_epoch, train_epoch
from stackformer.utils.device import move_to_device
from stackformer.utils.utils import is_main_process


class Engine:
    """Executes training and evaluation steps with robust batch handling and gradient accumulation.

    Simple explanation:
        The `Engine` wraps a `TrainingState` and handles the mechanics of running forward
        passes, computing loss, backpropagating gradients, scaling for AMP, stepping optimizers/schedulers,
        and tracking training metrics.

    Constructor args:
        state (TrainingState): Active training state container instance.
        grad_accum_steps (int, default=1): Number of micro-batches to accumulate before optimizer step.
        max_grad_norm (float | None, default=None): Threshold for gradient clipping.
        monitor (Any | None, default=None): Optional logger/monitor for live metric streaming.
        log_every (int, default=10): Metric logging interval (in steps).
        max_train_steps (int | None, default=None): Maximum global step cap for training termination.
        batch_parser (Callable | None, default=None): Custom batch unpacking function.
        compute_loss_fn (Callable | None, default=None): Custom loss calculation function.

    Example:
        >>> engine = Engine(state=training_state, grad_accum_steps=2)
        >>> metrics = engine._train_step(batch)
    """

    def __init__(
        self,
        state: Any,
        grad_accum_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        monitor: Optional[Any] = None,
        log_every: int = 10,
        max_train_steps: Optional[int] = None,
        batch_parser: Optional[Callable[[Any], Tuple[Any, Any]]] = None,
        compute_loss_fn: Optional[Callable[[Any, Any, Any], torch.Tensor]] = None,
    ) -> None:
        self.state = state
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.max_grad_norm = max_grad_norm
        self.max_train_steps = max_train_steps
        self.monitor = monitor
        self.live_monitoring = monitor is not None
        self.log_every = max(1, int(log_every))
        self.batch_parser = batch_parser
        self.compute_loss_fn = compute_loss_fn
        self.metrics = MetricTracker()
        self._accum_step = 0

        if self.state.model is not None:
            self.state.model.to(self.state.device)

    def train_one_epoch(self, dataloader: Any, epoch: int) -> None:
        """Run a full training epoch across the provided dataloader.

        Args:
            dataloader (Any): Training data loader iteration provider.
            epoch (int): Current epoch index.
        """
        self.metrics.reset()
        train_epoch(self, dataloader, epoch)

    def validate_one_epoch(self, dataloader: Any, epoch: int) -> None:
        """Run evaluation loop across validation dataloader.

        Args:
            dataloader (Any): Validation data loader iteration provider.
            epoch (int): Current epoch index.
        """
        eval_epoch(self, dataloader, epoch)
        scheduler = self.state.scheduler
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            val_loss = self.metrics.compute_epoch_metrics().get("val_loss")
            if val_loss is not None:
                scheduler.step(val_loss)

    def _train_step(self, batch: Any) -> Dict[str, Any]:
        """Execute a single training step (forward, loss calculation, backward, accumulation step).

        Args:
            batch (Any): Input batch tuple or dictionary.

        Returns:
            Dict[str, Any]: Step metric dictionary.
        """
        state = self.state
        if state.optimizer is None:
            raise ValueError("Optimizer is required for training steps.")

        self.metrics.start_step_timer()
        inputs, targets = self._prepare_batch(batch)
        scaler = state.scaler

        with self._get_autocast_context(scaler):
            outputs = self._forward_model(inputs)
            loss = self._compute_loss(outputs, targets)

        loss = loss / self.grad_accum_steps

        if scaler and scaler.is_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        self._accum_step += 1
        grad_norm = None

        if self._accum_step % self.grad_accum_steps == 0:
            if self.max_grad_norm is not None:
                grad_norm = self._clip_gradients()
            self._optimizer_step()
            self._scheduler_step()
            state.optimizer.zero_grad(set_to_none=True)
            state.step()
            self._accum_step = 0

        state.batch_idx += 1

        step_time = self.metrics.end_step_timer()
        loss_value = float(loss.item() * self.grad_accum_steps)
        self.metrics.update("loss", loss_value)

        lr = self.get_lr()
        if lr is not None:
            self.metrics.update("lr", lr)
        if step_time is not None:
            self.metrics.update("step_time", step_time)
        num_tokens = 0
        if torch.is_tensor(inputs) and inputs.dim() >= 2:
            num_tokens = inputs.numel()
        elif isinstance(inputs, dict):
            input_tensor = inputs.get("input_ids", inputs.get("x"))
            if torch.is_tensor(input_tensor) and input_tensor.dim() >= 2:
                num_tokens = input_tensor.numel()
        if num_tokens > 0:
            self.metrics.update_tokens(num_tokens)

        self.metrics.update_perplexity(loss_value)
        if grad_norm is not None:
            self.metrics.update("grad_norm", grad_norm)

        metrics = self.metrics.get_all(reduce_distributed=True)
        if state.global_step % self.log_every == 0:
            self._log_step(metrics)
        return metrics

    def _eval_step(self, batch: Any) -> Dict[str, Any]:
        """Execute a single non-optimizing evaluation step.

        Args:
            batch (Any): Input evaluation batch tuple or dictionary.

        Returns:
            Dict[str, Any]: Metric dictionary containing validation loss.
        """
        inputs, targets = self._prepare_batch(batch)
        scaler = self.state.scaler

        with self._get_autocast_context(scaler):
            outputs = self._forward_model(inputs)
            loss = self._compute_loss(outputs, targets)

        loss_value = float(loss.item())
        self.metrics.update("val_loss", loss_value)
        metrics = {"val_loss": loss_value}
        self._log_step(metrics)
        return metrics

    def _optimizer_step(self) -> None:
        state = self.state
        scaler = state.scaler
        if scaler and scaler.is_enabled:
            step_optimizer(state.optimizer, scaler)
            update_scaler(scaler)
        else:
            step_optimizer(state.optimizer, None)

    def _scheduler_step(self) -> None:
        scheduler = self.state.scheduler
        if scheduler is None:
            return
        if isinstance(scheduler, ReduceLROnPlateau):
            return
        scheduler.step()

    def _clip_gradients(self) -> float:
        state = self.state
        scaler = state.scaler
        if scaler and scaler.is_enabled:
            scaler.unscale_(state.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), self.max_grad_norm)
        return float(grad_norm)

    def _log_step(self, metrics: dict) -> None:
        if self.monitor is None or not is_main_process():
            return

        stable_metrics = dict(metrics)

        EXPECTED_KEYS = [
            "loss",
            "lr",
            "step_time",
            "tokens",
            "tokens_per_sec",
            "perplexity",
            "grad_norm",
        ]

        for key in EXPECTED_KEYS:
            stable_metrics.setdefault(key, None)

        self.monitor.log(stable_metrics)

    def get_lr(self) -> Optional[float]:
        """Return current learning rate from optimizer."""
        opt = self.state.optimizer
        if opt is None:
            return None
        return float(opt.param_groups[0]["lr"])

    def reached_max_train_steps(self) -> bool:
        """Check if global step limit has been reached."""
        if self.max_train_steps is None:
            return False
        return self.state.global_step >= int(self.max_train_steps)

    def _prepare_batch(self, batch: Any) -> Tuple[Any, Any]:
        if self.batch_parser is not None:
            inputs, targets = self.batch_parser(batch)
            return move_to_device(inputs, self.state.device), move_to_device(targets, self.state.device) if targets is not None else None

        if torch.is_tensor(batch):
            return move_to_device(batch, self.state.device), None

        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                inputs, targets = batch
                return move_to_device(inputs, self.state.device), move_to_device(targets, self.state.device) if targets is not None else None
            if len(batch) == 1:
                return move_to_device(batch[0], self.state.device), None

        if isinstance(batch, dict):
            targets = batch.get("targets", batch.get("labels"))
            model_inputs = {k: v for k, v in batch.items() if k not in {"targets", "labels"}}
            if "inputs" in model_inputs and len(model_inputs) == 1:
                model_inputs = model_inputs["inputs"]
            return (
                move_to_device(model_inputs, self.state.device),
                move_to_device(targets, self.state.device) if targets is not None else None,
            )

        return move_to_device(batch, self.state.device), None

    def _forward_model(self, inputs: Any) -> Any:
        if isinstance(inputs, dict):
            return self.state.model(**inputs)
        return self.state.model(inputs)

    def _compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        if self.compute_loss_fn is not None:
            return self.compute_loss_fn(outputs, targets, self.state)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        if isinstance(outputs, (list, tuple)) and outputs:
            if torch.is_tensor(outputs[0]) and outputs[0].numel() == 1:
                return outputs[0].squeeze()
            outputs = outputs[0]

        criterion = self.state.config.get("criterion")
        if criterion is None:
            if torch.is_tensor(outputs) and torch.is_tensor(targets):
                if (
                    targets.dtype in (torch.int64, torch.long)
                    and outputs.dim() == targets.dim() + 1
                    and outputs.shape[0] == targets.shape[0]
                ):
                    return F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                if outputs.shape == targets.shape:
                    return F.mse_loss(outputs, targets)

            raise ValueError("No criterion found. Pass criterion to Trainer or compute_loss_fn to Engine.")

        try:
            return criterion(outputs, targets)
        except TypeError:
            try:
                return criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-100)
            except TypeError:
                return criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

    @staticmethod
    def _get_autocast_context(scaler: Any) -> Any:
        if scaler and getattr(scaler, "is_enabled", False) and hasattr(scaler, "autocast"):
            return scaler.autocast()
        return nullcontext()