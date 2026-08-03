"""Feed-forward blocks used in StackFormer transformer layers.

Provides MLP variants for Transformer blocks:
- Standard 2-layer FFNs with activation choices (ReLU, LeakyReLU, GELU, Sigmoid, SiLU)
- Gated FFN variants (SwiGLU, GeGLU)

Canonical 2-layer FFN equation:
    FFN(x) = W2 * act(W1 * x + b1) + b2

Gated FFN equation (uses 3 weight matrices W_gate, W_up, W_down):
    y = W_down * (W_up * x ⊙ act(W_gate * x))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FF_ReLU(nn.Module):
    """Two-layer transformer FFN with ReLU activation.

    Computation:
        y = Dropout(W2 * Dropout(ReLU(W1 * x)))

    Constructor args:
        embed_dim (int): Input/output feature size ``C``.
        hidden_dim (int): Intermediate hidden size ``M``.
        dropout (float, default=0.0): Dropout probability in hidden and output projections.
        device (torch.device | str, default='cpu'): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, T, C)``.

    Returns:
        torch.Tensor: Output tensor of shape ``(B, T, C)``.

    Example:
        >>> ff = FF_ReLU(embed_dim=512, hidden_dim=2048, dropout=0.1)
        >>> x = torch.randn(4, 32, 512)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([4, 32, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.relu = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return self.relu(x)


class FF_LeakyReLU(nn.Module):
    """Two-layer FFN with LeakyReLU nonlinearity.

    Constructor args:
        embed_dim (int): Input/output size ``C``.
        hidden_dim (int): Intermediate size ``M``.
        dropout (float, default=0.0): Dropout probability.
        negative_slope (float, default=0.1): LeakyReLU slope for negative values.
        device (torch.device | str, default='cpu'): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, T, C)``.

    Returns:
        torch.Tensor: Output tensor of shape ``(B, T, C)``.

    Example:
        >>> ff = FF_LeakyReLU(embed_dim=256, hidden_dim=1024, negative_slope=0.01)
        >>> x = torch.randn(2, 16, 256)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([2, 16, 256])
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        negative_slope: float = 0.1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.l_relu = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return self.l_relu(x)


class FF_GELU(nn.Module):
    """Two-layer FFN with GELU activation.

    Constructor args:
        embed_dim (int): Input/output size ``C``.
        hidden_dim (int): Intermediate size ``M``.
        dropout (float, default=0.0): Dropout probability.
        device (torch.device | str, default='cpu'): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, T, C)``.

    Returns:
        torch.Tensor: Output tensor of shape ``(B, T, C)``.

    Example:
        >>> ff = FF_GELU(embed_dim=768, hidden_dim=3072)
        >>> x = torch.randn(2, 128, 768)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([2, 128, 768])
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.gelu = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return self.gelu(x)


class FF_Sigmoid(nn.Module):
    """Two-layer FFN with Sigmoid activation.

    Constructor args:
        embed_dim (int): Input/output size ``C``.
        hidden_dim (int): Intermediate size ``M``.
        dropout (float, default=0.0): Dropout probability.
        device (torch.device | str, default='cpu'): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, T, C)``.

    Returns:
        torch.Tensor: Output tensor of shape ``(B, T, C)``.

    Example:
        >>> ff = FF_Sigmoid(embed_dim=128, hidden_dim=512)
        >>> x = torch.randn(3, 20, 128)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([3, 20, 128])
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.sigmoid = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return self.sigmoid(x)


class FF_SiLU(nn.Module):
    """Two-layer FFN with SiLU/Swish activation.

    Constructor args:
        embed_dim (int): Input/output size ``C``.
        hidden_dim (int): Intermediate size ``M``.
        dropout (float, default=0.0): Dropout probability.
        device (torch.device | str, default='cpu'): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, T, C)``.

    Returns:
        torch.Tensor: Output tensor of shape ``(B, T, C)``.

    Example:
        >>> ff = FF_SiLU(embed_dim=512, hidden_dim=2048, dropout=0.1)
        >>> x = torch.randn(1, 64, 512)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([1, 64, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.silu = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return self.silu(x)


class FF_SwiGLU(nn.Module):
    """Gated FFN with SwiGLU activation (Swish-Gated Linear Unit).

    Computation:
        y = W_down * (W_up * x ⊙ SiLU(W_gate * x))

    Parameter note:
        SwiGLU constructs 3 linear projection matrices (W_gate, W_up, W_down), resulting in 50%
        more parameters than a standard 2-layer FFN for the same `hidden_dim`. To match standard FFN
        parameter counts, set `hidden_dim = int(2/3 * 4 * embed_dim)`.

    Constructor args:
        embed_dim (int): Input/output size ``C``.
        hidden_dim (int): Inner gated branch dimension ``M``.
        dropout (float, default=0.0): Dropout probability after output projection.
        device (torch.device | str, default='cpu'): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, T, C)``.

    Returns:
        torch.Tensor: Output tensor of shape ``(B, T, C)``.

    Example:
        >>> ff = FF_SwiGLU(embed_dim=1024, hidden_dim=2048, dropout=0.1)
        >>> x = torch.randn(2, 32, 1024)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([2, 32, 1024])
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.W_gate = nn.Linear(embed_dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.W_up = nn.Linear(embed_dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.W_down = nn.Linear(hidden_dim, embed_dim, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return self.dropout(self.W_down(F.silu(self.W_gate(x)) * self.W_up(x)))


class FF_GeGLU(nn.Module):
    """Gated FFN with GeGLU activation (GELU-Gated Linear Unit).

    Computation:
        y = W_down * (W_up * x ⊙ GELU(W_gate * x))

    Parameter note:
        GeGLU constructs 3 linear projection matrices (via 2*hidden_dim projection + W_down),
        resulting in 50% more parameters than a standard 2-layer FFN for the same `hidden_dim`.

    Constructor args:
        embed_dim (int): Input/output size ``C``.
        hidden_dim (int): Inner gated branch dimension ``M``.
        dropout (float, default=0.0): Dropout probability.
        device (torch.device | str, default='cpu'): Target device.
        dtype (torch.dtype, default=torch.float32): Tensor data type.

    Forward args:
        x (torch.Tensor): Input tensor of shape ``(B, T, C)``.

    Returns:
        torch.Tensor: Output tensor of shape ``(B, T, C)``.

    Example:
        >>> ff = FF_GeGLU(embed_dim=512, hidden_dim=1536)
        >>> x = torch.randn(2, 40, 512)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([2, 40, 512])
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim * 2, device=device, dtype=dtype)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x_proj = self.linear1(x)  # (B, T, 2*M)
        x1, x2 = x_proj.chunk(2, dim=-1)  # (B, T, M) each
        x = x2 * F.gelu(x1)  # (B, T, M)
        x = self.dropout1(x)
        x = self.linear2(x)  # (B, T, C)
        x = self.dropout2(x)
        return x


# New canonical class names (Step 3 rename)
FeedForwardReLU = FF_ReLU
FeedForwardLeakyReLU = FF_LeakyReLU
FeedForwardGELU = FF_GELU
FeedForwardSigmoid = FF_Sigmoid
FeedForwardSiLU = FF_SiLU
FeedForwardSwiGLU = FF_SwiGLU
FeedForwardGeGLU = FF_GeGLU
