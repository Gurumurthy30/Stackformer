import torch
from tests._test_utils import _checkpoint

from stackformer.modules.attention import Multi_Head_Attention
from stackformer.modules.feed_forward import FF_GELU
from stackformer.modules.position_embedding import AbsolutePositionEmbedding


def test_embedding_position_attention_ff_pipeline_end_to_end(torch_device):
    _checkpoint("test_embedding_position_attention_ff_pipeline_end_to_end setup", device=torch_device)
    torch.manual_seed(0)
    tokens = torch.randint(0, 32, (2, 6), device=torch_device)

    token_emb = torch.nn.Embedding(32, 8, device=torch_device)
    pos_emb = AbsolutePositionEmbedding(seq_len=6, embed_dim=8, device=torch_device)
    attn = Multi_Head_Attention(embed_dim=8, num_heads=2, dropout=0.0, device=torch_device)
    ff = FF_GELU(embed_dim=8, hidden_dim=16, dropout=0.0, device=torch_device)

    _checkpoint("Executing pipeline operations")
    x = token_emb(tokens)
    x = x + pos_emb(tokens)
    x = attn(x)
    x = ff(x)

    _checkpoint("Asserting output shape and finiteness", x_shape=x.shape)
    assert x.shape == (2, 6, 8)
    assert torch.isfinite(x).all()
    assert torch.abs(x).mean() < 100

    x.mean().backward()
    _checkpoint("Checking embedding gradients")
    assert token_emb.weight.grad is not None
