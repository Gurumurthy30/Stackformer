import math

import pytest
import torch

from stackformer.modules.position_embedding import (
    AbsolutePositionEmbedding,
    RoPE,
    SinusoidalPositionalEmbedding,
)
from tests._test_utils import _checkpoint


@pytest.mark.parametrize("batch,seq", [(1, 4), (2, 6)])
def test_absolute_position_embedding_shapes_and_stability(batch, seq, torch_device):
    _checkpoint(
        "test_absolute_position_embedding_shapes_and_stability setup",
        batch=batch,
        seq=seq,
        device=torch_device,
    )
    torch.manual_seed(0)
    layer = AbsolutePositionEmbedding(seq_len=8, embed_dim=8, device=torch_device)
    x = torch.randint(0, 16, (batch, seq), device=torch_device)

    out = layer(x)
    _checkpoint("Asserting AbsolutePositionEmbedding output shape", out_shape=out.shape)
    assert out.shape == (batch, seq, 8)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("batch,seq", [(1, 4), (2, 6)])
def test_sinusoidal_position_embedding_shapes_and_stability(batch, seq, torch_device):
    _checkpoint(
        "test_sinusoidal_position_embedding_shapes_and_stability setup",
        batch=batch,
        seq=seq,
        device=torch_device,
    )
    torch.manual_seed(0)
    layer = SinusoidalPositionalEmbedding(seq_len=8, embed_dim=8, device=torch_device)
    x = torch.randn(batch, seq, 8, device=torch_device)

    out = layer(x)
    _checkpoint("Asserting SinusoidalPositionalEmbedding output shape", out_shape=out.shape)
    assert out.shape == (batch, seq, 8)
    assert torch.isfinite(out).all()
    # Exact value check on pe buffer (#43)
    expected_row0 = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], device=torch_device)
    assert torch.allclose(layer.pe[0], expected_row0, atol=1e-6)
    expected_row1 = torch.tensor(
        [
            math.sin(1.0),
            math.cos(1.0),
            math.sin(0.1),
            math.cos(0.1),
            math.sin(0.01),
            math.cos(0.01),
            math.sin(0.001),
            math.cos(0.001),
        ],
        device=torch_device,
    )
    assert torch.allclose(layer.pe[1], expected_row1, atol=1e-5)


def test_rope_compatibility_with_attention_shape_and_stability(torch_device):
    _checkpoint(
        "test_rope_compatibility_with_attention_shape_and_stability setup", device=torch_device
    )
    torch.manual_seed(0)
    rope = RoPE(head_dim=4, seq_len=8, device=torch_device)
    x = torch.randn(2, 2, 6, 4, device=torch_device)

    out = rope(x)
    _checkpoint("Asserting RoPE output shape", out_shape=out.shape)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_rope_requires_even_head_dim(torch_device):
    _checkpoint("test_rope_requires_even_head_dim setup")
    with pytest.raises((ValueError, AssertionError)):
        RoPE(head_dim=3, seq_len=6, device=torch_device)


def test_absolute_position_embedding_fails_when_seq_is_too_long(torch_device):
    _checkpoint("test_absolute_position_embedding_fails_when_seq_is_too_long setup")
    layer = AbsolutePositionEmbedding(seq_len=4, embed_dim=8, device=torch_device)
    with pytest.raises((ValueError, IndexError)):
        layer(torch.randint(0, 10, (2, 5), device=torch_device))


def test_rope_fails_for_odd_input_last_dim(torch_device):
    _checkpoint("test_rope_fails_for_odd_input_last_dim setup")
    rope = RoPE(head_dim=4, seq_len=8, device=torch_device)
    with pytest.raises((ValueError, AssertionError)):
        rope(torch.randn(1, 2, 4, 3, device=torch_device))


def test_sinusoidal_requires_even_embed_dim(torch_device):
    """Cover M-04 issue verifying SinusoidalPositionalEmbedding validation for odd embed_dim."""
    _checkpoint("test_sinusoidal_requires_even_embed_dim setup")
    with pytest.raises(ValueError):
        SinusoidalPositionalEmbedding(seq_len=8, embed_dim=7, device=torch_device)


def test_absolute_position_embedding_token_invariance(torch_device):
    """Finding #30: AbsolutePositionEmbedding.forward(x) must produce standard arange positions."""
    layer = AbsolutePositionEmbedding(seq_len=16, embed_dim=8, device=torch_device)
    # Two token sequences of length 4 with different token IDs, both with IDs < seq_len
    x1 = torch.tensor([[0, 1, 2, 3]], device=torch_device)
    x2 = torch.tensor([[5, 4, 2, 1]], device=torch_device)
    pos1 = layer(x1)
    pos2 = layer(x2)
    # Both must return the positional embeddings for positions 0, 1, 2, 3
    assert torch.allclose(pos1, pos2), "AbsolutePositionEmbedding must not index by token ID"


def test_absolute_position_embedding_forward_ids(torch_device):
    """Finding #30: forward_ids explicitly indexes by position IDs."""
    layer = AbsolutePositionEmbedding(seq_len=16, embed_dim=8, device=torch_device)
    p_ids = torch.tensor([[2, 3, 4]], device=torch_device)
    out = layer.forward_ids(p_ids)
    expected = layer.embedding(p_ids)
    assert torch.allclose(out, expected)
