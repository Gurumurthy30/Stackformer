import math
import torch
from tests._test_utils import _checkpoint
from stackformer.attention_engine.masking.dense import make_mask

BATCH = 2
SEQ = 6
EMB = 8


def _manual_masked_attention(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    q = x
    k = x
    v = x

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(x.size(-1))
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def test_causal_mask_prevents_future_token_leakage(torch_device):
    _checkpoint("test_causal_mask_prevents_future_token_leakage setup", device=torch_device)

    x1 = torch.randn(BATCH, SEQ, EMB, device=torch_device)
    x2 = x1.clone()
    x2[:, -1] += 5.0

    _checkpoint("Creating causal mask", seq_len=SEQ)
    causal_mask = make_mask(["causal"], seq_len=SEQ, device=torch_device)

    _checkpoint("Computing manual masked attention")
    out1 = _manual_masked_attention(x1, causal_mask)
    out2 = _manual_masked_attention(x2, causal_mask)

    _checkpoint("Asserting no leakage into earlier positions", out1_shape=out1.shape)
    assert out1.shape == out2.shape == (BATCH, SEQ, EMB)
    assert torch.isfinite(out1).all() and torch.isfinite(out2).all()
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6)