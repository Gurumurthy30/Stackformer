import torch
from tests._test_utils import _checkpoint
from stackformer.modules.attention import (
    Group_query_Attention,
    Multi_Head_Attention,
    kv_cache_group_query,
    kv_cache_multihead,
)


def _copy_qkv_proj_weights(src, dst):
    """Copy weights between modules that both use q_proj + fused kv_proj
    (Multi_Head_Attention <-> kv_cache_multihead, and
     Group_query_Attention <-> kv_cache_group_query).

    Both sides have matching q_proj and kv_proj shapes for these pairs, so a
    direct copy works with no splitting/reshaping.
    """
    with torch.no_grad():
        dst.q_proj.weight.copy_(src.q_proj.weight)
        dst.kv_proj.weight.copy_(src.kv_proj.weight)
        dst.out_proj.weight.copy_(src.out_proj.weight)
        if src.q_proj.bias is not None and dst.q_proj.bias is not None:
            dst.q_proj.bias.copy_(src.q_proj.bias)
        if src.kv_proj.bias is not None and dst.kv_proj.bias is not None:
            dst.kv_proj.bias.copy_(src.kv_proj.bias)
        if src.out_proj.bias is not None and dst.out_proj.bias is not None:
            dst.out_proj.bias.copy_(src.out_proj.bias)


def test_kv_cache_multihead_matches_full_attention_for_last_token(torch_device):
    _checkpoint("test_kv_cache_multihead_matches_full_attention_for_last_token setup", device=torch_device)
    x = torch.randn(2, 6, 8, device=torch_device)

    # kv_cache_multihead has q_proj + kv_proj (not q/k/v_proj), so we need a
    # Multi_Head_Attention-shaped source with the same layout. Multi_Head_Attention
    # itself uses a single fused qkv_proj, so we build the "full" reference using
    # kv_cache_multihead's own forward with a large enough cache instead of trying
    # to copy weights from Multi_Head_Attention (whose qkv_proj layout differs).
    full = kv_cache_multihead(embed_dim=8, num_heads=2, batch_size=2, kv_seq_len=6, dropout=0.0, device=torch_device)
    cached = kv_cache_multihead(embed_dim=8, num_heads=2, batch_size=2, kv_seq_len=6, dropout=0.0, device=torch_device)
    _copy_qkv_proj_weights(full, cached)

    _checkpoint("Running full forward pass (single call covering all tokens)")
    full_out = full(x, start_pos=0, rope=False)

    _checkpoint("Stepping cached forward pass token-by-token")
    step_out = None
    for t in range(x.size(1)):
        step_out = cached(x[:, t : t + 1], start_pos=t, rope=False)

    _checkpoint("Asserting step output matches full attention last token", step_out_shape=step_out.shape)
    assert step_out is not None
    assert step_out.shape == (2, 1, 8)
    assert torch.allclose(step_out.squeeze(1), full_out[:, -1, :], atol=1e-5)


def test_kv_cache_group_query_matches_full_attention_for_last_token(torch_device):
    _checkpoint("test_kv_cache_group_query_matches_full_attention_for_last_token setup", device=torch_device)
    x = torch.randn(2, 6, 8, device=torch_device)

    full = Group_query_Attention(
        embed_dim=8, num_query_heads=2, num_kv_heads=1, dropout=0.0, mask_type=["causal"], device=torch_device
    )
    cached = kv_cache_group_query(
        embed_dim=8, num_query_heads=2, num_kv_heads=1, batch_size=2, kv_seq_len=6, dropout=0.0, device=torch_device
    )
    _copy_qkv_proj_weights(full, cached)

    _checkpoint("Running full GQA forward pass")
    full_out = full(x)

    _checkpoint("Stepping cached GQA forward pass token-by-token")
    step_out = None
    for t in range(x.size(1)):
        step_out = cached(x[:, t : t + 1], start_pos=t, rope=False)

    _checkpoint("Asserting step output matches full GQA last token", step_out_shape=step_out.shape)
    assert step_out is not None
    assert step_out.shape == (2, 1, 8)
    assert torch.allclose(step_out.squeeze(1), full_out[:, -1, :], atol=1e-5)
