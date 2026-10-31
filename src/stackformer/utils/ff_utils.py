"""Feed-forward network (FFN) dimension helper utilities.

Provides `hidden_dim_helper` for calculating hidden dimensions for SwiGLU and GeGLU layers.
"""

from __future__ import annotations


def hidden_dim_helper(
    embed_dim: int,
    ffn_multiplier: float = 4.0,
    multiple_of: int = 256,
) -> int:
    """Compute the recommended hidden dimension for SwiGLU/GeGLU FFNs.

    Gated feed-forward networks use three projection matrices instead of two.
    To keep the parameter count approximately equal to a standard Transformer
    FFN with `hidden_dim = ffn_multiplier * embed_dim`, the hidden dimension
    is reduced by a factor of 2/3 and rounded to a multiple of `multiple_of`.

    Args:
        embed_dim (int): Model embedding dimension.
        ffn_multiplier (float, default=4.0): Standard FFN expansion ratio.
        multiple_of (int, default=256): Round result up to a multiple of this value.

    Returns:
        int: Recommended hidden dimension value.

    Example:
        >>> hidden_dim_helper(4096)
        11008
        >>> hidden_dim_helper(768)
        2048
    """
    hidden_dim = int((2.0 / 3.0) * ffn_multiplier * embed_dim)

    if multiple_of > 1:
        hidden_dim = ((hidden_dim + multiple_of - 1) // multiple_of) * multiple_of

    return hidden_dim