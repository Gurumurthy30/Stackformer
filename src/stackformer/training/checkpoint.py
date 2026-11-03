"""Checkpoint management for Stackformer Trainer.

Provides atomic, single-process synchronous checkpointing, pruning, best-model tracking,
and pointer-based resumption for training workflows.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from stackformer.training.state import TrainState
from stackformer.utils.checkpoint import load_model_weights, save_model_weights

try:
    import numpy as np
except ImportError:
    np = None

try:
    import safetensors.torch
except ImportError:
    safetensors = None


def unwrap_model(model: nn.Module) -> nn.Module:
    """Recursively unwrap compile, DDP, and wrapper modules."""
    while hasattr(model, "_orig_mod") or hasattr(model, "module"):
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif hasattr(model, "module"):
            model = model.module
    return model


def clean_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Clean model state dict keys, stripping _orig_mod. and module. prefixes."""
    cleaned = {}
    for k, v in raw_state.items():
        clean_k = k
        while "_orig_mod." in clean_k:
            clean_k = clean_k.replace("_orig_mod.", "")
        while clean_k.startswith("module."):
            clean_k = clean_k[7:]
        cleaned[clean_k] = v
    return cleaned


class CheckpointManager:
    """Manages training checkpoints, metadata, pruning, and state restoration.

    Layout:
        <checkpoint_dir>/
            epoch_001_step_00100/
                state.pt           (model, optim, scheduler, scaler, rng, state, config)
                model.safetensors  (weights-only for standalone loading)
                meta.json          (human-readable training metrics and metadata)
            best/
                model.safetensors
                meta.json
                state.pt           (optional, if save_best_state=True)
            latest                 (text file containing newest checkpoint directory name)

    Args:
        checkpoint_dir: Root directory for saving checkpoints (default: "./checkpoints").
        keep_last_n: Number of recent checkpoints to retain (older are pruned).
        save_best: Whether to track and store the best performing checkpoint.
        monitor: Metric name to evaluate for best-model selection.
        mode: Direction of metric optimization ("min" or "max").
        save_best_state: Whether to also save full optimizer/state in the best/ folder.
        async_checkpoint: Placeholder for future asynchronous checkpointing.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "./checkpoints",
        keep_last_n: int = 3,
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_state: bool = False,
        async_checkpoint: bool = False,
        use_dcp: bool = False,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.keep_last_n = max(1, keep_last_n)
        self.save_best_enabled = save_best
        self.monitor = monitor
        self.mode = mode.lower()
        self.save_best_state = save_best_state
        self.async_checkpoint = async_checkpoint
        self.use_dcp = use_dcp

    def _get_rng_state(self) -> dict[str, Any]:
        """Collect current random number generator states across all engines."""
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state() if np is not None else None,
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        return rng

    def _set_rng_state(self, rng_state: dict[str, Any]) -> None:
        """Restore random number generator states."""
        if not rng_state:
            return
        if "python" in rng_state and rng_state["python"] is not None:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state and rng_state["numpy"] is not None and np is not None:
            np.random.set_state(rng_state["numpy"])
        if "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
            torch.set_rng_state(rng_state["torch_cpu"])
        if "torch_cuda" in rng_state and rng_state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    def save(
        self,
        model: nn.Module,
        train_state: TrainState,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        config: Optional[Any] = None,
        metrics: Optional[dict[str, float]] = None,
    ) -> Path:
        """Atomically save training state, weights, and metadata.

        Args:
            model: PyTorch model.
            train_state: Dynamic training state (epoch, step counters, best metric).
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler instance.
            scaler: GradScaler instance if mixed precision is used.
            config: TrainConfig or config dictionary.
            metrics: Optional evaluated metrics dictionary.

        Returns:
            Path: Path to saved checkpoint directory.
        """
        dir_name = f"epoch_{train_state.epoch + 1:03d}_step_{train_state.global_step:05d}"
        target_dir = self.checkpoint_dir / dir_name

        from stackformer.distributed.utils import is_distributed_initialized, is_rank_zero
        is_dist = is_distributed_initialized() and torch.distributed.get_world_size() > 1

        if self.use_dcp and is_dist:
            # Multi-process DCP: all ranks write shards to target_dir/dcp concurrently
            target_dir.mkdir(parents=True, exist_ok=True)
            dcp_dir = target_dir / "dcp"
            from stackformer.distributed.dcp import save_dcp
            model_state = clean_state_dict(model.state_dict())
            state_dict = {
                "model": model_state,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
                "rng": self._get_rng_state(),
                "train_state": train_state.to_dict(),
                "config": config.to_dict() if hasattr(config, "to_dict") else config,
            }
            save_dcp(state_dict, dcp_dir)

            if is_rank_zero():
                if safetensors is not None:
                    safetensors.torch.save_model(unwrap_model(model), str(target_dir / "model.safetensors"))
                meta = {
                    "step": train_state.global_step,
                    "epoch": train_state.epoch,
                    "step_in_epoch": train_state.step_in_epoch,
                    "total_micro_batches": train_state.total_micro_batches,
                    "best_metric": train_state.best_metric,
                    "metrics": metrics or {},
                    "torch_version": torch.__version__,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with open(target_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                self._update_latest_pointer(dir_name)
                self.prune()

            torch.distributed.barrier()
            return target_dir

        temp_dir = self.checkpoint_dir / f".tmp_{dir_name}_{os.getpid()}"

        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            model_state = clean_state_dict(model.state_dict())
            # 1. state.pt or DCP
            state_dict = {
                "model": model_state,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
                "rng": self._get_rng_state(),
                "train_state": train_state.to_dict(),
                "config": config.to_dict() if hasattr(config, "to_dict") else config,
            }
            if self.use_dcp:
                dcp_dir = temp_dir / "dcp"
                from stackformer.distributed.dcp import save_dcp
                save_dcp(state_dict, dcp_dir)
            else:
                torch.save(state_dict, temp_dir / "state.pt")

            # 2. model.safetensors
            if safetensors is not None:
                safetensors.torch.save_model(unwrap_model(model), str(temp_dir / "model.safetensors"))
            else:
                warnings.warn(
                    "safetensors is not installed; saved weights to 'model.pt' instead. "
                    "Install safetensors via: pip install safetensors",
                    UserWarning,
                    stacklevel=2,
                )
                torch.save(model_state, temp_dir / "model.pt")

            # 3. meta.json
            meta = {
                "step": train_state.global_step,
                "epoch": train_state.epoch,
                "step_in_epoch": train_state.step_in_epoch,
                "total_micro_batches": train_state.total_micro_batches,
                "best_metric": train_state.best_metric,
                "metrics": metrics or {},
                "torch_version": torch.__version__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(temp_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # Atomic directory placement
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            temp_dir.rename(target_dir)

            # Update latest pointer atomically
            self._update_latest_pointer(dir_name)

            # Prune older checkpoints
            self.prune()

            return target_dir
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _update_latest_pointer(self, dir_name: str) -> None:
        """Write directory name to a temporary file then replace 'latest' pointer atomically."""
        temp_latest = self.checkpoint_dir / f".tmp_latest_{os.getpid()}"
        with open(temp_latest, "w", encoding="utf-8") as f:
            f.write(dir_name.strip() + "\n")
        temp_latest.replace(self.checkpoint_dir / "latest")

    def latest(self) -> Optional[Path]:
        """Retrieve path to the most recent checkpoint directory, or None if none exist."""
        latest_file = self.checkpoint_dir / "latest"
        if latest_file.exists():
            try:
                name = latest_file.read_text(encoding="utf-8").strip()
                p = self.checkpoint_dir / name
                if p.is_dir() and ((p / "state.pt").exists() or (p / "dcp").exists() or (p / "model.safetensors").exists()):
                    return p
            except Exception:
                pass

        # Fallback: scan directories
        checkpoints = self._list_checkpoints()
        if checkpoints:
            return checkpoints[-1][1]
        return None

    def _list_checkpoints(self) -> list[tuple[tuple[int, int], Path]]:
        """List all valid checkpoint directories sorted ascending by (epoch, step)."""
        pattern = re.compile(r"^epoch_(\d+)_step_(\d+)$")
        results = []
        if not self.checkpoint_dir.exists():
            return []

        for item in self.checkpoint_dir.iterdir():
            if item.is_dir():
                m = pattern.match(item.name)
                if m and ((item / "state.pt").exists() or (item / "dcp").exists()):
                    epoch = int(m.group(1))
                    step = int(m.group(2))
                    results.append(((epoch, step), item))

        results.sort(key=lambda x: x[0])
        return results

    def prune(self) -> None:
        """Prune older checkpoints exceeding keep_last_n. Never deletes best/ directory."""
        checkpoints = self._list_checkpoints()
        if len(checkpoints) > self.keep_last_n:
            to_delete = checkpoints[: len(checkpoints) - self.keep_last_n]
            for _, path in to_delete:
                shutil.rmtree(path, ignore_errors=True)

    def load(
        self,
        path: str | Path | None = "auto",
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[dict[str, Any]]:
        """Load checkpoint state dictionary from specific path or 'auto' for latest.

        Args:
            path: File path, directory path, or "auto" (default: "auto").
            model: Optional model instance for in-place DCP loading.
            optimizer: Optional optimizer instance for in-place DCP loading.

        Returns:
            dict[str, Any] | None: Deserialized checkpoint dictionary, or None if starting fresh.
        """
        if path is None or (isinstance(path, str) and path.lower() == "auto"):
            ckpt_dir = self.latest()
            if ckpt_dir is None:
                return None
            target_path = ckpt_dir
        else:
            p = Path(path)
            if not p.is_absolute() and not p.exists() and (self.checkpoint_dir / p).exists():
                p = self.checkpoint_dir / p
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found at: {path}")
            target_path = p

        if target_path.is_dir():
            if (target_path / "dcp").is_dir() or (target_path / ".metadata").exists():
                from stackformer.distributed.dcp import load_dcp

                dcp_dir = target_path / "dcp" if (target_path / "dcp").is_dir() else target_path
                state = {}
                if model is not None:
                    state["model"] = model.state_dict()
                if optimizer is not None:
                    state["optimizer"] = optimizer.state_dict()
                load_dcp(state, dcp_dir)
                return state

            target_file = target_path / "state.pt"
            if not target_file.exists():
                raise FileNotFoundError(f"Checkpoint directory {target_path} does not contain 'state.pt' or 'dcp'")
        else:
            target_file = target_path

        return torch.load(str(target_file), map_location="cpu", weights_only=False)

    def save_best(
        self,
        model: nn.Module,
        train_state: TrainState,
        metrics: Optional[dict[str, float]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> Path:
        """Save best-performing checkpoint into the dedicated 'best/' directory.

        Args:
            model: PyTorch model.
            train_state: Training state.
            metrics: Evaluation metrics.
            optimizer: Optional optimizer for full state save.
            scheduler: Optional scheduler for full state save.
            scaler: Optional scaler for full state save.
            config: Optional config for full state save.

        Returns:
            Path: Path to best/ directory.
        """
        best_dir = self.checkpoint_dir / "best"
        temp_dir = self.checkpoint_dir / f".tmp_best_{os.getpid()}"

        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            model_state = clean_state_dict(model.state_dict())
            # 1. model.safetensors
            if safetensors is not None:
                safetensors.torch.save_model(unwrap_model(model), str(temp_dir / "model.safetensors"))
            else:
                warnings.warn(
                    "safetensors is not installed; saved weights to 'model.pt' instead. "
                    "Install safetensors via: pip install safetensors",
                    UserWarning,
                    stacklevel=2,
                )
                torch.save(model_state, temp_dir / "model.pt")

            # 2. meta.json
            meta = {
                "step": train_state.global_step,
                "epoch": train_state.epoch,
                "best_metric": train_state.best_metric,
                "metrics": metrics or {},
                "torch_version": torch.__version__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(temp_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # 3. optional full state.pt
            if self.save_best_state:
                state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer is not None else None,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
                    "rng": self._get_rng_state(),
                    "train_state": train_state.to_dict(),
                    "config": config.to_dict() if hasattr(config, "to_dict") else config,
                }
                torch.save(state_dict, temp_dir / "state.pt")

            # Atomic placement
            if best_dir.exists():
                shutil.rmtree(best_dir, ignore_errors=True)
            temp_dir.rename(best_dir)

            return best_dir
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def load_best(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        """Load best-performing weights from best/ directory into model.

        Args:
            model: Target neural network module.
            device: Target compute device.
        """
        best_dir = self.checkpoint_dir / "best"
        if not best_dir.exists():
            raise FileNotFoundError(f"Best checkpoint directory not found at: {best_dir}")

        st_file = best_dir / "model.safetensors"
        pt_file = best_dir / "model.pt"
        state_file = best_dir / "state.pt"

        if st_file.exists():
            load_model_weights(model, st_file, device=device)
        elif pt_file.exists():
            load_model_weights(model, pt_file, device=device)
        elif state_file.exists():
            ckpt = torch.load(str(state_file), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
        else:
            raise FileNotFoundError(f"No weights file found in best checkpoint directory: {best_dir}")

