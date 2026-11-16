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


def test_sinusoidal_exact_formula_reference(torch_device):
    """Finding #47: Sinusoidal table matches textbook formula across all positions and dimensions."""
    seq_len = 16
    embed_dim = 16
    layer = SinusoidalPositionalEmbedding(seq_len=seq_len, embed_dim=embed_dim, device=torch_device)

    positions = torch.arange(seq_len, dtype=torch.float32, device=torch_device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32, device=torch_device)
        * -(math.log(10000.0) / embed_dim)
    )
    expected = torch.zeros(seq_len, embed_dim, device=torch_device)
    expected[:, 0::2] = torch.sin(positions * div_term)
    expected[:, 1::2] = torch.cos(positions * div_term)

    assert torch.allclose(layer.pe, expected, atol=1e-6)


def test_rope_rotation_matrix_reference(torch_device):
    """Finding #47: RoPE matches exact 2D rotation matrix: position 0 is identity, position t rotates by frequencies."""
    head_dim = 4
    seq_len = 8
    rope = RoPE(head_dim=head_dim, seq_len=seq_len, device=torch_device)

    # 1. Position 0 must be an exact identity operation
    x0 = torch.randn(1, 1, 1, head_dim, device=torch_device)
    out0 = rope(x0)
    assert torch.allclose(out0, x0, atol=1e-6), "RoPE at position 0 must act as identity"

    # 2. Test rotation of 2D pairs at position t against explicit 2D rotation matrix
    x = torch.randn(1, 1, seq_len, head_dim, device=torch_device)
    out = rope(x)

    for t in range(seq_len):
        for k in range(head_dim // 2):
            freq = 1.0 / (10000.0 ** ((2 * k) / head_dim))
            theta_t = t * freq
            cos_t = math.cos(theta_t)
            sin_t = math.sin(theta_t)

            u = x[0, 0, t, 2 * k].item()
            v = x[0, 0, t, 2 * k + 1].item()

            expected_u = u * cos_t - v * sin_t
            expected_v = u * sin_t + v * cos_t

            actual_u = out[0, 0, t, 2 * k].item()
            actual_v = out[0, 0, t, 2 * k + 1].item()

            assert math.isclose(actual_u, expected_u, rel_tol=1e-5, abs_tol=1e-5)
            assert math.isclose(actual_v, expected_v, rel_tol=1e-5, abs_tol=1e-5)


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
