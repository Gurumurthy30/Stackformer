"""Parameter initialization strategies for neural network layers and transformers.

Provides functional tensor in-place initializers wrapping and extending PyTorch's
`torch.nn.init` functions with sensible transformer defaults.
"""

from __future__ import annotations

import torch
import torch.nn.init as init


def normal_init_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02) -> torch.Tensor:
    """Initialize tensor with normal (Gaussian) distribution.

    Args:
        tensor (torch.Tensor): Target tensor to initialize.
        mean (float, default=0.0): Mean of the normal distribution.
        std (float, default=0.02): Standard deviation of the normal distribution.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    return init.normal_(tensor, mean=mean, std=std)


def uniform_init_(tensor: torch.Tensor, a: float = -0.02, b: float = 0.02) -> torch.Tensor:
    """Initialize tensor with uniform distribution over interval [a, b).

    Args:
        tensor (torch.Tensor): Target tensor to initialize.
        a (float, default=-0.02): Lower bound.
        b (float, default=0.02): Upper bound.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    return init.uniform_(tensor, a=a, b=b)


def trunc_normal_init_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.02,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Initialize tensor with truncated normal distribution.

    Values outside [a, b] (expressed in standard deviations or raw bounds) are redrawn.

    Args:
        tensor (torch.Tensor): Target tensor to initialize.
        mean (float, default=0.0): Mean of normal distribution.
        std (float, default=0.02): Standard deviation.
        a (float, default=-2.0): Minimum cutoff value.
        b (float, default=2.0): Maximum cutoff value.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    return init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Initialize tensor with Glorot / Xavier uniform distribution.

    Args:
        tensor (torch.Tensor): Target tensor to initialize (ndim >= 2).
        gain (float, default=1.0): Scaling factor.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    return init.xavier_uniform_(tensor, gain=gain)


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Initialize tensor with Glorot / Xavier normal distribution.

    Args:
        tensor (torch.Tensor): Target tensor to initialize (ndim >= 2).
        gain (float, default=1.0): Scaling factor.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    return init.xavier_normal_(tensor, gain=gain)


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    """Initialize tensor with He / Kaiming uniform distribution.

    Args:
        tensor (torch.Tensor): Target tensor to initialize.
        a (float, default=0.0): Negative slope of rectifier.
        mode (str, default='fan_in'): 'fan_in' or 'fan_out'.
        nonlinearity (str, default='leaky_relu'): Non-linear function name.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    return init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> torch.Tensor:
    """Initialize tensor with He / Kaiming normal distribution.

    Args:
        tensor (torch.Tensor): Target tensor to initialize.
        a (float, default=0.0): Negative slope of rectifier.
        mode (str, default='fan_in'): 'fan_in' or 'fan_out'.
        nonlinearity (str, default='leaky_relu'): Non-linear function name.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    return init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)


def orthogonal_init_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Initialize tensor with orthogonal matrix initialization.

    Args:
        tensor (torch.Tensor): Target 2D or higher tensor.
        gain (float, default=1.0): Multiplicative factor.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    return init.orthogonal_(tensor, gain=gain)


def zeros_init_(tensor: torch.Tensor) -> torch.Tensor:
    """Initialize tensor with constant zeros.

    Args:
        tensor (torch.Tensor): Target tensor to zero out.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    return init.zeros_(tensor)


def ones_init_(tensor: torch.Tensor) -> torch.Tensor:
    """Initialize tensor with constant ones.

    Args:
        tensor (torch.Tensor): Target tensor to fill with 1.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    return init.ones_(tensor)
