"""Weight initialization subsystem for StackFormer architectures.

Provides mathematical strategies, modular initializers, and traversal helpers:
- `Initializer`: Module-wide initialization manager
- `init_weights`: Functional initialization helper
- Mathematical in-place strategies: `normal_init_`, `trunc_normal_init_`, `xavier_uniform_`,
  `xavier_normal_`, `kaiming_uniform_`, `kaiming_normal_`, `orthogonal_init_`, `zeros_init_`,
  `ones_init_`, `uniform_init_`.
"""

from .initializer import Initializer, init_weights
from .strategies import (
    kaiming_normal_,
    kaiming_uniform_,
    normal_init_,
    ones_init_,
    orthogonal_init_,
    trunc_normal_init_,
    uniform_init_,
    xavier_normal_,
    xavier_uniform_,
    zeros_init_,
)

__all__ = [
    "Initializer",
    "init_weights",
    "normal_init_",
    "trunc_normal_init_",
    "uniform_init_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_init_",
    "zeros_init_",
    "ones_init_",
]
