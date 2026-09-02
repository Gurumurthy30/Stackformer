"""Checkpoint save and load utilities for StackFormer models.

Provides three complementary checkpoint mechanisms:
1. SafeTensors (`save` / `load`): Full consolidated model weights file safe for export.
2. PyTorch State (`save` / `load`): Full optimizer, scheduler, scaler, and training state dictionary.
3. Distributed Checkpoint (`save_sharded` / `load_sharded`): High-performance sharded DCP checkpointing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file

from stackformer.utils.utils import is_main_process, print_once

try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        get_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
        set_state_dict,
    )
    from torch.distributed.checkpoint.stateful import Stateful

    _DCP_AVAILABLE = True
except ImportError:  # torch < 2.2: no DCP state-dict helpers available
    _DCP_AVAILABLE = False

    class Stateful:  # type: ignore[no-redef]
        """No-op stand-in for Stateful on PyTorch versions prior to 2.2."""


class _AppState(Stateful):
    """Wraps model and optimizer for Distributed Checkpoint (DCP) sharded state handling.

    Args:
        model (torch.nn.Module): Target model instance.
        optimizer (torch.optim.Optimizer | None, default=None): Optional optimizer instance.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        """Return model and optimizer state dictionaries via DCP helper."""
        model_state, optim_state = get_state_dict(self.model, self.optimizer)
        return {"model": model_state, "optim": optim_state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore model and optimizer state dictionaries via DCP helper."""
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


class CheckpointManager:
    """Manages saving and loading of model weights and training execution state.

    Constructor args:
        output_dir (str): Parent directory path to store checkpoint files.
        device (str | torch.device, default="cpu"): Target compute device for loaded state.
    """

    def __init__(
        self,
        output_dir: str,
        device: str | torch.device = "cpu",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: Dict[str, Any], name: str = "latest") -> Dict[str, str] | None:
        """Save consolidated SafeTensors weights and PyTorch training state.

        Args:
            state (Dict[str, Any]): Dictionary containing 'model', optional 'optimizer', 'scheduler', etc.
            name (str, default="latest"): Checkpoint file identifier suffix.

        Returns:
            Dict[str, str] | None: Dictionary of saved file paths on rank 0, or None on worker ranks.
        """
        model = state.get("model")
        if model is None:
            raise ValueError("state['model'] is required.")

        model_state, optimizer_state = self._gather_full_state(model, state.get("optimizer"))

        if not is_main_process():
            return None

        weights_path = self._weights_path(name)
        state_path = self._state_path(name)

        self._safe_save_safetensors(model_state, weights_path)

        training_state = self._build_train_state(state, optimizer_state=optimizer_state)
        self._safe_torch_save(training_state, state_path)

        print_once(f"[Checkpoint] Weights saved -> {weights_path}")
        print_once(f"[Checkpoint] State saved   -> {state_path}")

        return {
            "weights": str(weights_path),
            "state": str(state_path),
        }

    def load(
        self,
        name: str,
        state: Dict[str, Any],
        broadcast_from_rank0: bool = False,
    ) -> Dict[str, Any]:
        """Restore model weights and training progress metrics from a saved checkpoint.

        Args:
            name (str): Checkpoint identifier name (e.g. "latest", "best").
            state (Dict[str, Any]): Dictionary containing target 'model' and optional 'optimizer'.
            broadcast_from_rank0 (bool, default=False): Broadcast rank 0 weights across workers.

        Returns:
            Dict[str, Any]: Restored training metadata (epoch, global_step, batch_idx, config, seed).
        """
        weights_path = self._weights_path(name)
        state_path = self._state_path(name)

        if not weights_path.exists():
            raise FileNotFoundError(weights_path)
        if not state_path.exists():
            raise FileNotFoundError(state_path)

        model = state.get("model")
        if model is None:
            raise ValueError("state['model'] is required.")

        broadcast = broadcast_from_rank0 and _DCP_AVAILABLE
        should_read_weights = (not broadcast) or is_main_process()
        model_state = load_file(str(weights_path), device="cpu") if should_read_weights else {}

        self._scatter_full_model_state(model, model_state, broadcast_from_rank0=broadcast)

        checkpoint = torch.load(
            state_path,
            map_location=self.device,
            weights_only=True,
        )

        self._scatter_optimizer_state(model, state.get("optimizer"), checkpoint.get("optimizer_state_dict"))
        self._maybe_load_state_dict(state.get("scheduler"), checkpoint.get("scheduler_state_dict"), "scheduler")
        self._maybe_load_state_dict(state.get("scaler"), checkpoint.get("scaler_state_dict"), "scaler")

        print_once(f"[Checkpoint] Loaded <- {name}")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "batch_idx": checkpoint.get("batch_idx", 0),
            "config": checkpoint.get("config", {}),
            "seed": checkpoint.get("seed"),
            "jit_model_path": checkpoint.get("jit_model_path"),
        }

    def save_sharded(self, state: Dict[str, Any], name: str = "latest") -> Dict[str, str]:
        """Save per-rank sharded checkpoint directory via Distributed Checkpoint (DCP).

        Args:
            state (Dict[str, Any]): State dictionary containing 'model' and optional 'optimizer'.
            name (str, default="latest"): Checkpoint name suffix.

        Returns:
            Dict[str, str]: Dictionary containing path to sharded directory and metadata file.
        """
        self._require_dcp()

        model = state.get("model")
        if model is None:
            raise ValueError("state['model'] is required.")

        app_state = _AppState(model, state.get("optimizer"))

        ckpt_dir = self._sharded_dir(name)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        dcp.save({"app": app_state}, checkpoint_id=str(ckpt_dir))

        result = {"sharded_dir": str(ckpt_dir)}

        if is_main_process():
            meta_path = self._sharded_meta_path(name)
            training_state = self._build_train_state(state, include_optimizer=False)
            self._safe_torch_save(training_state, meta_path)
            result["state"] = str(meta_path)
            print_once(f"[Checkpoint] Sharded weights -> {ckpt_dir}")
            print_once(f"[Checkpoint] State saved      -> {meta_path}")

        return result

    def load_sharded(self, name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Restore sharded model and optimizer state via Distributed Checkpoint (DCP).

        Args:
            name (str): Checkpoint identifier name.
            state (Dict[str, Any]): State dictionary containing target 'model' and optional 'optimizer'.

        Returns:
            Dict[str, Any]: Restored training metadata.
        """
        self._require_dcp()

        ckpt_dir = self._sharded_dir(name)
        meta_path = self._sharded_meta_path(name)

        if not ckpt_dir.exists():
            raise FileNotFoundError(ckpt_dir)
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        model = state.get("model")
        if model is None:
            raise ValueError("state['model'] is required.")

        app_state = _AppState(model, state.get("optimizer"))
        dcp.load({"app": app_state}, checkpoint_id=str(ckpt_dir))

        checkpoint = torch.load(
            meta_path,
            map_location=self.device,
            weights_only=True,
        )

        self._maybe_load_state_dict(state.get("scheduler"), checkpoint.get("scheduler_state_dict"), "scheduler")
        self._maybe_load_state_dict(state.get("scaler"), checkpoint.get("scaler_state_dict"), "scaler")

        print_once(f"[Checkpoint] Loaded (sharded) <- {name}")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "batch_idx": checkpoint.get("batch_idx", 0),
            "config": checkpoint.get("config", {}),
            "seed": checkpoint.get("seed"),
            "jit_model_path": checkpoint.get("jit_model_path"),
        }

    def save_jit_model(self, model: torch.nn.Module, name: str = "latest") -> str | None:
        """Export TorchScript compiled model artifact to disk.

        Args:
            model (torch.nn.Module): Model instance to compile and save.
            name (str, default="latest"): Output filename identifier.

        Returns:
            str | None: Path to saved TorchScript file (or None on non-main process).
        """
        if not is_main_process():
            return None

        model = self._unwrap_model(model)
        model.eval()

        path = self._jit_path(name)
        tmp = path.with_suffix(".tmp")

        try:
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, tmp)
            os.replace(tmp, path)

            print_once(f"[Checkpoint] TorchScript -> {path}")
            return str(path)

        except Exception as exc:
            if tmp.exists():
                tmp.unlink()

            raise RuntimeError(f"TorchScript save failed: {exc}") from exc

    def load_jit_model(self, path: str) -> torch.jit.ScriptModule:
        """Load compiled TorchScript module from file.

        Args:
            path (str): Filepath to TorchScript module.

        Returns:
            torch.jit.ScriptModule: Loaded TorchScript module in eval mode.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        model = torch.jit.load(
            str(file_path),
            map_location=self.device,
        )

        model.eval()
        return model

    def _weights_path(self, name: str) -> Path:
        return self.output_dir / f"checkpoint_{name}.safetensors"

    def _state_path(self, name: str) -> Path:
        return self.output_dir / f"checkpoint_{name}.pt"

    def _jit_path(self, name: str) -> Path:
        return self.output_dir / f"checkpoint_{name}.jit.pt"

    def _sharded_dir(self, name: str) -> Path:
        return self.output_dir / f"checkpoint_{name}_dcp"

    def _sharded_meta_path(self, name: str) -> Path:
        return self._sharded_dir(name) / "train_state.pt"

    @staticmethod
    def _gather_full_state(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        if _DCP_AVAILABLE:
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)

            model_state = get_model_state_dict(model=model, options=options)
            model_state = {
                k: (v.contiguous() if isinstance(v, torch.Tensor) else v)
                for k, v in model_state.items()
            }

            optimizer_state = (
                get_optimizer_state_dict(model=model, optimizers=optimizer, options=options)
                if optimizer is not None
                else None
            )
            return model_state, optimizer_state

        unwrapped = CheckpointManager._unwrap_model(model)
        model_state = {
            k: v.detach().cpu().contiguous() for k, v in unwrapped.state_dict().items()
        }
        optimizer_state = CheckpointManager._safe_state_dict(optimizer)
        return model_state, optimizer_state

    def _scatter_full_model_state(
        self,
        model: torch.nn.Module,
        model_state: Dict[str, Any],
        broadcast_from_rank0: bool = False,
    ) -> None:
        if _DCP_AVAILABLE and broadcast_from_rank0:
            options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=broadcast_from_rank0)
            set_model_state_dict(model=model, model_state_dict=model_state, options=options)
            return

        if not model_state:
            return

        unwrapped = self._unwrap_model(model)
        unwrapped.load_state_dict(model_state)
        unwrapped.to(self.device)

    def _scatter_optimizer_state(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        optimizer_state: Optional[Dict[str, Any]],
    ) -> None:
        if optimizer is None or optimizer_state is None:
            return

        if _DCP_AVAILABLE:
            try:
                options = StateDictOptions(full_state_dict=True)
                set_optimizer_state_dict(
                    model=model,
                    optimizers=optimizer,
                    optim_state_dict=optimizer_state,
                    options=options,
                )
            except Exception as exc:
                print_once(f"[Checkpoint] Warning: failed to restore optimizer: {exc}")
            return

        self._maybe_load_state_dict(optimizer, optimizer_state, "optimizer")

    def _build_train_state(
        self,
        state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        include_optimizer: bool = True,
    ) -> Dict[str, Any]:
        train_state = {
            "scheduler_state_dict": self._safe_state_dict(state.get("scheduler")),
            "scaler_state_dict": self._safe_state_dict(state.get("scaler")),
            "epoch": state.get("epoch", 0),
            "global_step": state.get("global_step", 0),
            "batch_idx": state.get("batch_idx", 0),
            "config": state.get("config", {}),
            "seed": state.get("seed", 42),
            "jit_model_path": state.get("jit_model_path"),
        }
        if include_optimizer:
            train_state["optimizer_state_dict"] = optimizer_state
        return train_state

    @staticmethod
    def _require_dcp() -> None:
        if not _DCP_AVAILABLE:
            raise RuntimeError(
                "Sharded (DCP) checkpointing needs torch>=2.2 "
                "(torch.distributed.checkpoint.state_dict). Detected "
                f"torch=={torch.__version__}."
            )

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _safe_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None or not hasattr(obj, "state_dict"):
            return None

        try:
            return obj.state_dict()
        except Exception:
            return None

    @staticmethod
    def _maybe_load_state_dict(
        obj: Any,
        state_dict: Optional[Dict[str, Any]],
        label: str,
    ) -> None:
        if obj is None or state_dict is None:
            return

        if not hasattr(obj, "load_state_dict"):
            return

        try:
            obj.load_state_dict(state_dict)

        except Exception as exc:
            print_once(f"[Checkpoint] Warning: failed to restore {label}: {exc}")

    @staticmethod
    def _safe_torch_save(
        obj: Dict[str, Any],
        path: Path,
    ) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")

        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)

        finally:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _safe_save_safetensors(
        tensors: Dict[str, torch.Tensor],
        path: Path,
    ) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")

        try:
            save_file(tensors, str(tmp))
            os.replace(tmp, path)

        finally:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass