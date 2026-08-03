import pytest
import torch
from tests._test_utils import _checkpoint
from stackformer.models.transformer import Transformer
from stackformer.modules.Attention import Multi_Head_Attention
from stackformer.modules.Feed_forward import FF_ReLU
from stackformer.modules.position_embedding import SinusoidalPositionalEmbedding

@pytest.mark.parametrize("batch,seq,embed_dim,heads", [(1, 4, 8, 1), (2, 6, 16, 4)])
def test_transformer_forward_shape_finite_and_backward(batch, seq, embed_dim, heads, torch_device):
    
    _checkpoint("test_transformer_forward_shape_finite_and_backward setup", batch=batch, seq=seq, device=torch_device)
    torch.manual_seed(0)
    vocab_size = 32
    x = torch.randint(0, vocab_size, (batch, seq), device=torch_device)

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=heads,
        dropout=0.0,
        hidden_dim=embed_dim * 2,
        encoder_layers=1,
        decoder_layers=1,
        seq_len=seq,
        device=torch_device,
    )
    _checkpoint("Executing seq2seq Transformer forward pass")
    out = model(x, x)
    _checkpoint("Asserting Transformer output shape and finiteness", out_shape=out.shape)
    assert out.shape == (batch, seq, vocab_size)
    assert torch.isfinite(out).all()
    assert torch.abs(out).mean() < 100

    out.mean().backward()
    _checkpoint("Checking parameter gradients")
    assert model.token_emb.weight.grad is not None
    assert model.out_proj.weight.grad is not None


def test_module_interoperability_embedding_pos_attention_ff_pipeline(torch_device):
    _checkpoint("test_module_interoperability_embedding_pos_attention_ff_pipeline setup", device=torch_device)
    torch.manual_seed(0)
    batch, seq, vocab_size, embed_dim = 2, 6, 32, 8
    token_embedding = torch.nn.Embedding(vocab_size, embed_dim, device=torch_device)
    pos_embedding = SinusoidalPositionalEmbedding(seq_len=seq, embed_dim=embed_dim, device=torch_device)
    attention = Multi_Head_Attention(embed_dim=embed_dim, num_heads=2, mask_type=["no"], dropout=0.0, device=torch_device)
    ff = FF_ReLU(embed_dim=embed_dim, hidden_dim=16, dropout=0.0, device=torch_device)

    ids = torch.randint(0, vocab_size, (batch, seq), device=torch_device)
    _checkpoint("Running pipeline steps")
    x = token_embedding(ids)
    x = x + pos_embedding(x)
    x = attention(x)
    x = ff(x)

    _checkpoint("Asserting pipeline output shape", x_shape=x.shape)
    assert x.shape == (batch, seq, embed_dim)
    assert torch.isfinite(x).all()

    x.mean().backward()
    assert token_embedding.weight.grad is not None


def test_transformer_invalid_head_setup_raises(torch_device):
    _checkpoint("test_transformer_invalid_head_setup_raises setup")
    with pytest.raises((AssertionError, ValueError)):
        Transformer(
            vocab_size=20,
            embed_dim=10,
            num_heads=3,
            dropout=0.0,
            hidden_dim=16,
            encoder_layers=1,
            decoder_layers=1,
            seq_len=4,
            device=torch_device,
        )
