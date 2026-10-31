"""Base model interface providing unified parameter lifecycle and checkpointing.

Defines the `BaseModel` abstract class from which all StackFormer neural network
architectures inherit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from stackformer.initialization import Initializer
from stackformer.utils.checkpoint import load_model_weights, save_model_weights
from stackformer.utils.pretrained import download_and_load_pretrained


def resolve_model_config(
    first_arg: Any = None,
    explicit_config: Any = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Helper to resolve configuration and kwargs when initializing models.

    Supports initializing with `Model(config)`, `Model(config=config)`, or
    `Model(param1=val1, param2=val2, ...)`.
    """
    config = explicit_config
    resolved = dict(kwargs)

    if first_arg is not None:
        if isinstance(first_arg, (int, float, str, bool)):
            pass
        elif hasattr(first_arg, "__dict__") or hasattr(first_arg, "__dataclass_fields__") or isinstance(first_arg, dict):
            config = first_arg

    if config is not None:
        if isinstance(config, dict):
            for k, v in config.items():
                resolved[k] = v
        else:
            for attr in dir(config):
                if not attr.startswith("_"):
                    try:
                        val = getattr(config, attr)
                        if not callable(val):
                            resolved[attr] = val
                    except Exception:
                        pass

    try:
        current_ctx_device = torch.empty(0).device
        if current_ctx_device.type == "meta":
            resolved["device"] = current_ctx_device
    except Exception:
        pass

    return config, resolved


class BaseModel(nn.Module):
    """Abstract base class for all StackFormer model architectures.

    Provides standardized lifecycle methods for large-scale language and vision models:
    - Module definition and registration in `__init__`
    - Parameter initialization via `create_weights(method=...)`
    - Meta-device tensor allocation via `materialize(device=...)`
    - Checkpoint loading from file or remote repository via `load_weights(path=...)`
    - Checkpoint serialization to PyTorch or SafeTensors format via `save_weights(path)`
    - Parameter counting via `parameter_count(trainable_only=...)`

    Example:
        >>> from stackformer.config import ModelConfig
        >>> from stackformer.models import GPT2
        >>> config = ModelConfig(vocab_size=1000, embed_dim=128, num_layers=2, num_heads=4, seq_len=64, hidden_dim=256)
        >>> model = GPT2(config)
        >>> model.create_weights()
        GPT2(...)
        >>> count = model.parameter_count()
        >>> model.save_weights("checkpoint.safetensors")
        >>> model.load_weights("checkpoint.safetensors")
        GPT2(...)

    Args:
        config (Any | None, default=None): Model configuration object or dictionary.
        **kwargs: Additional configuration parameters stored on `self.config`.
    """

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        """Architecture-specific default parameter initialization hook.

        Subclasses may override this method to provide custom initialization logic
        (such as special scaling for residual projections or truncated normal distributions).
        Default behavior applies normal distribution with std=0.02 to Linear and Embedding layers,
        ones to normalization weights, and zeros to biases.

        Args:
            module (nn.Module): Module currently visited during recursion.
        """
        if isinstance(module, nn.Linear):
            if not module.weight.is_meta:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not module.bias.is_meta:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if not module.weight.is_meta:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None and not module.weight.is_meta:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            if module.weight is not None and not module.weight.is_meta:
                nn.init.ones_(module.weight)
            if module.bias is not None and not module.bias.is_meta:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "weight") and hasattr(module, "eps") and not hasattr(module, "bias"):
            if module.weight is not None and not module.weight.is_meta:
                nn.init.ones_(module.weight)

    def create_weights(self, method: str = "default", **kwargs: Any) -> BaseModel:
        """Initialize model parameters using default or specified initialization strategy.

        Args:
            method (str, default='default'): Initialization strategy.
                If 'default', invokes `self._init_weights()` across all submodules.
                Otherwise invokes `Initializer(strategy=method, **kwargs)`.
            **kwargs: Extra parameters passed to the `Initializer`.

        Returns:
            BaseModel: Self for fluent method chaining.
        """
        if method == "default":
            self.apply(self._init_weights)
        else:
            Initializer(strategy=method, **kwargs).apply(self)
        return self

    def materialize(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> BaseModel:
        """Materialize meta-device parameters into real allocated memory on target device.

        When models are instantiated within `with torch.device("meta"):`, no tensor memory
        is allocated. `materialize()` allocates physical memory for all parameters and
        buffers so weights can subsequently be loaded or initialized.

        Args:
            device (torch.device | str, default='cpu'): Compute device for memory allocation.
            dtype (torch.dtype | None, default=None): Optional tensor data type.

        Returns:
            BaseModel: Self for fluent method chaining.
        """
        is_meta = any(p.is_meta for p in self.parameters()) or any(b.is_meta for b in self.buffers())
        if is_meta:
            self.to_empty(device=device, recurse=True)
        else:
            self.to(device=device)

        if dtype is not None:
            self.to(dtype=dtype)

        return self

    def save_weights(self, path: str | Path) -> None:
        """Save model state dictionary to disk.

        Automatically selects SafeTensors format if the file extension is '.safetensors',
        otherwise saves as a PyTorch checkpoint.

        Args:
            path (str | Path): Destination file path.
        """
        save_model_weights(self, path)

    def load_weights(
        self,
        path: str | Path | None = None,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> BaseModel:
        """Load model state dictionary from local path or fetch pretrained weights.

        Args:
            path (str | Path | None, default=None): Checkpoint file path (.pt or .safetensors).
                If None, calls `download_and_load_pretrained()` to fetch canonical weights.
            strict (bool, default=True): Enforce strict state dict matching.
            device (torch.device | str, default='cpu'): Target device for loaded weights.

        Returns:
            BaseModel: Self for fluent method chaining.
        """
        if path is None:
            download_and_load_pretrained(self)
        else:
            load_model_weights(self, path=path, strict=strict, device=device)
        return self

    def parameter_count(self, trainable_only: bool = False) -> int:
        """Count total or trainable parameters in the model.

        Args:
            trainable_only (bool, default=False): If True, count only parameters with
                `requires_grad == True`.

        Returns:
            int: Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
