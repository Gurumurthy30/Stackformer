import torch

def _grad_safe_splice(cached: torch.Tensor, live: torch.Tensor, start_pos: int, end_pos: int) -> torch.Tensor:
    """Shared by every policy: when gradients are enabled, an in-place
    write into a persistent buffer would corrupt the autograd graph.
    Detach+clone the historical prefix and splice the live
    (still-grad-tracked) chunk back in for the current span."""
    if not torch.is_grad_enabled():
        return cached
    out = cached.detach().clone()
    out[:, :, start_pos:end_pos] = live
    return out