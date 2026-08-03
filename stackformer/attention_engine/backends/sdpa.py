import torch
import torch.nn.functional as F

def _run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
) -> torch.Tensor:
    """Wrapper around PyTorch's scaled dot-product attention (SDPA).

    Args:
        q (torch.Tensor): Query tensor of shape `(B, H, T, D)`.
        k (torch.Tensor): Key tensor of shape `(B, H, T, D)`.
        v (torch.Tensor): Value tensor of shape `(B, H, T, D)`.
        attn_mask (torch.Tensor | None): Attention mask tensor or None.
        dropout_p (float): Attention dropout probability.

    Returns:
        torch.Tensor: Output tensor of shape `(B, H, T, D)`.
    """
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )
