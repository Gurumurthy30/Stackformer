import torch

from stackformer.modules.attention import Multi_Head_Attention, Self_Attention
from tests._test_utils import _checkpoint


def test_self_attention_equivalent_to_single_head_mha_when_weights_are_shared(torch_device):
    _checkpoint("test_self_attention_equivalent_to_single_head_mha setup", device=torch_device)
    x = torch.randn(2, 6, 8, requires_grad=True, device=torch_device)

    _checkpoint("Instantiating Self_Attention and single-head Multi_Head_Attention")
    sa = Self_Attention(embed_dim=8, dropout=0.0, qkv_bias=True, mask_type=["causal"], device=torch_device)
    mha = Multi_Head_Attention(
        embed_dim=8, num_heads=1, dropout=0.0, qkv_bias=True, mask_type=["causal"], device=torch_device
    )

    _checkpoint("Copying projection weights from SA to MHA")
    with torch.no_grad():
        # Both classes use a single fused qkv_proj (embed_dim -> 3*embed_dim),
        # so weights can be copied directly with no splitting/reshaping needed.
        mha.qkv_proj.weight.copy_(sa.qkv_proj.weight)
        mha.out_proj.weight.copy_(sa.out_proj.weight)
        if mha.qkv_proj.bias is not None and sa.qkv_proj.bias is not None:
            mha.qkv_proj.bias.copy_(sa.qkv_proj.bias)
        if mha.out_proj.bias is not None and sa.out_proj.bias is not None:
            mha.out_proj.bias.copy_(sa.out_proj.bias)

    _checkpoint("Executing forward passes")
    out_sa = sa(x)
    out_mha = mha(x)

    _checkpoint("Asserting outputs match within tolerance", sa_shape=out_sa.shape, mha_shape=out_mha.shape)
    assert out_sa.shape == out_mha.shape == x.shape
    assert torch.isfinite(out_sa).all() and torch.isfinite(out_mha).all()
    assert torch.allclose(out_sa, out_mha, atol=1e-5)
