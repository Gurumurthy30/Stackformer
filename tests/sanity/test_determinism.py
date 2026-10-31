import torch
from tests._test_utils import _checkpoint
from stackformer import GPT_1
from stackformer.modules.attention import Multi_Head_Attention
from stackformer.modules.position_embedding import AbsolutePositionEmbedding, RoPE, SinusoidalPositionalEmbedding


def test_attention_determinism_fixed_seed(torch_device):
    _checkpoint("test_attention_determinism_fixed_seed setup", device=torch_device)
    x = torch.randn(2, 6, 8, device=torch_device)

    _checkpoint("Instantiating Multi_Head_Attention m1 & m2", shape=x.shape, dtype=x.dtype)
    torch.manual_seed(0)
    m1 = Multi_Head_Attention(embed_dim=8, num_heads=2, dropout=0.0, device=torch_device)
    y1 = m1(x)

    torch.manual_seed(0)
    m2 = Multi_Head_Attention(embed_dim=8, num_heads=2, dropout=0.0, device=torch_device)
    y2 = m2(x)

    _checkpoint("Asserting determinism", y1_shape=y1.shape, y2_shape=y2.shape)
    assert torch.allclose(y1, y2)


def test_position_embeddings_determinism_fixed_seed(torch_device):
    _checkpoint("test_position_embeddings_determinism_fixed_seed setup", device=torch_device)
    ids = torch.randint(0, 8, (2, 6), device=torch_device)

    _checkpoint("Testing AbsolutePositionEmbedding determinism")
    torch.manual_seed(0)
    abs1 = AbsolutePositionEmbedding(seq_len=6, embed_dim=8, device=torch_device)
    a1 = abs1(ids)

    torch.manual_seed(0)
    abs2 = AbsolutePositionEmbedding(seq_len=6, embed_dim=8, device=torch_device)
    a2 = abs2(ids)

    s = SinusoidalPositionalEmbedding(seq_len=6, embed_dim=8, device=torch_device)
    r = RoPE(head_dim=4, seq_len=6)
    x = torch.randn(2, 2, 6, 4, device=torch_device)

    _checkpoint("Asserting position embedding outputs match", a1_shape=a1.shape)
    assert torch.allclose(a1, a2)
    assert torch.allclose(s(ids), s(ids))
    assert torch.allclose(r(x), r(x))


def test_gpt_determinism_fixed_seed(torch_device):
    _checkpoint("test_gpt_determinism_fixed_seed setup", device=torch_device)
    x = torch.randint(0, 32, (2, 6), device=torch_device)

    _checkpoint("Instantiating GPT_1 models for determinism test")
    torch.manual_seed(0)
    m1 = GPT_1(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16, device=torch_device)
    y1 = m1(x)

    torch.manual_seed(0)
    m2 = GPT_1(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16, device=torch_device)
    y2 = m2(x)

    _checkpoint("Asserting GPT output determinism", y1_shape=y1.shape, y2_shape=y2.shape)
    assert torch.allclose(y1, y2)
