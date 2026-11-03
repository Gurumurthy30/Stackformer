"""Configurable parameter initializer for neural network modules and architectures.

Provides the `Initializer` class and `init_weights` helper to traverse PyTorch
module hierarchies and apply named weight initialization strategies.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from stackformer.initialization import strategies


class Initializer:
    """Modular parameter initialization manager for neural network models.

    Supports configurable initialization strategies including Normal, Truncated Normal,
    Xavier/Glorot (uniform and normal), He/Kaiming (uniform and normal), and Orthogonal.
    Automatically applies zero-initialization to biases and one-initialization to
    normalization weights.

    Example:
        >>> import torch.nn as nn
        >>> from stackformer.initialization import Initializer
        >>> model = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256))
        >>> init = Initializer(strategy="xavier_uniform")
        >>> init.apply(model)
        Sequential(...)

    Args:
        strategy (str, default="normal"): Name of initialization strategy.
            Supported: "normal", "trunc_normal", "xavier_uniform", "xavier_normal",
            "kaiming_uniform", "kaiming_normal", "orthogonal", "uniform".
        std (float, default=0.02): Standard deviation for normal/trunc_normal.
        mean (float, default=0.0): Mean for normal/trunc_normal.
        gain (float, default=1.0): Multiplicative factor for xavier/orthogonal.
        **kwargs: Additional hyperparameters passed to strategy functions.
    """

    STRATEGY_MAP: dict[str, Callable[..., torch.Tensor]] = {
        "normal": strategies.normal_init_,
        "gaussian": strategies.normal_init_,
        "trunc_normal": strategies.trunc_normal_init_,
        "truncated_normal": strategies.trunc_normal_init_,
        "uniform": strategies.uniform_init_,
        "xavier": strategies.xavier_uniform_,
        "xavier_uniform": strategies.xavier_uniform_,
        "xavier_normal": strategies.xavier_normal_,
        "kaiming": strategies.kaiming_uniform_,
        "kaiming_uniform": strategies.kaiming_uniform_,
        "kaiming_normal": strategies.kaiming_normal_,
        "orthogonal": strategies.orthogonal_init_,
    }

    def __init__(
        self,
        strategy: str = "normal",
        std: float = 0.02,
        mean: float = 0.0,
        gain: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.strategy_name = strategy.lower().strip()
        if self.strategy_name not in self.STRATEGY_MAP:
            valid = list(self.STRATEGY_MAP.keys())
            raise ValueError(f"Unknown initialization strategy '{strategy}'. Supported strategies: {valid}")
        self.strategy_fn = self.STRATEGY_MAP[self.strategy_name]
        self.std = std
        self.mean = mean
        self.gain = gain
        self.kwargs = kwargs

    def _init_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Initialize a single weight tensor using configured strategy."""
        if tensor.is_meta:
            return tensor

        name = self.strategy_name
        if name in ("normal", "gaussian"):
            return strategies.normal_init_(tensor, mean=self.mean, std=self.std)
        if name in ("trunc_normal", "truncated_normal"):
            return strategies.trunc_normal_init_(
                tensor,
                mean=self.mean,
                std=self.std,
                a=self.kwargs.get("a", -2.0 * self.std),
                b=self.kwargs.get("b", 2.0 * self.std),
            )
        if name == "uniform":
            return strategies.uniform_init_(
                tensor,
                a=self.kwargs.get("a", -self.std),
                b=self.kwargs.get("b", self.std),
            )
        if name in ("xavier", "xavier_uniform"):
            if tensor.ndim >= 2:
                return strategies.xavier_uniform_(tensor, gain=self.gain)
            return strategies.normal_init_(tensor, mean=self.mean, std=self.std)
        if name == "xavier_normal":
            if tensor.ndim >= 2:
                return strategies.xavier_normal_(tensor, gain=self.gain)
            return strategies.normal_init_(tensor, mean=self.mean, std=self.std)
        if name in ("kaiming", "kaiming_uniform"):
            if tensor.ndim >= 2:
                return strategies.kaiming_uniform_(
                    tensor,
                    a=self.kwargs.get("a", 0.0),
                    mode=self.kwargs.get("mode", "fan_in"),
                    nonlinearity=self.kwargs.get("nonlinearity", "leaky_relu"),
                )
            return strategies.normal_init_(tensor, mean=self.mean, std=self.std)
        if name == "kaiming_normal":
            if tensor.ndim >= 2:
                return strategies.kaiming_normal_(
                    tensor,
                    a=self.kwargs.get("a", 0.0),
                    mode=self.kwargs.get("mode", "fan_in"),
                    nonlinearity=self.kwargs.get("nonlinearity", "leaky_relu"),
                )
            return strategies.normal_init_(tensor, mean=self.mean, std=self.std)
        if name == "orthogonal":
            if tensor.ndim >= 2:
                return strategies.orthogonal_init_(tensor, gain=self.gain)
            return strategies.normal_init_(tensor, mean=self.mean, std=self.std)

        return self.strategy_fn(tensor)

    def _init_module(self, module: nn.Module) -> None:
        """Module visitor applying weights and biases rules."""
        # Linear layers
        if isinstance(module, nn.Linear):
            self._init_tensor(module.weight)
            if module.bias is not None and not module.bias.is_meta:
                strategies.zeros_init_(module.bias)

        # Embedding layers
        elif isinstance(module, nn.Embedding):
            self._init_tensor(module.weight)
            if module.padding_idx is not None and not module.weight.is_meta:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

        # Convolutional layers
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self._init_tensor(module.weight)
            if module.bias is not None and not module.bias.is_meta:
                strategies.zeros_init_(module.bias)

        # Normalization layers (LayerNorm, RMSNorm, BatchNorm, GroupNorm)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            if module.weight is not None and not module.weight.is_meta:
                strategies.ones_init_(module.weight)
            if module.bias is not None and not module.bias.is_meta:
                strategies.zeros_init_(module.bias)
        elif hasattr(module, "weight") and hasattr(module, "eps") and not hasattr(module, "bias"):
            # Covers RMSNorm variants without direct nn.LayerNorm inheritance
            if module.weight is not None and not module.weight.is_meta:
                strategies.ones_init_(module.weight)

    def apply(self, module: nn.Module) -> nn.Module:
        """Apply initialization across all submodules in module hierarchy.

        Args:
            module (nn.Module): Target PyTorch module or model.

        Returns:
            nn.Module: The initialized module.
        """
        module.apply(self._init_module)
        return module


def init_weights(
    module: nn.Module,
    strategy: str = "normal",
    std: float = 0.02,
    mean: float = 0.0,
    gain: float = 1.0,
    **kwargs: Any,
) -> nn.Module:
    """Convenience helper to initialize a module hierarchy with a specified strategy.

    Args:
        module (nn.Module): PyTorch module to initialize.
        strategy (str, default='normal'): Strategy name.
        std (float, default=0.02): Standard deviation for normal distributions.
        mean (float, default=0.0): Mean for normal distributions.
        gain (float, default=1.0): Gain factor for xavier/orthogonal.
        **kwargs: Strategy-specific parameters.

    Returns:
        nn.Module: Initialized module.
    """
    initializer = Initializer(strategy=strategy, std=std, mean=mean, gain=gain, **kwargs)
    return initializer.apply(module)
