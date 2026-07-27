import pytest
import torch
from tests._test_utils import _checkpoint
from stackformer.modules.Attention import kv_cache_group_query, kv_cache_multihead


@pytest.mark.parametrize("batch", [1, 2])
def test_kv_cache_multihead_updates_cache_and_stays_stable(batch, torch_device):
    _checkpoint("test_kv_cache_multihead_updates_cache_and_stays_stable setup", batch=batch, device=torch_device)
    layer = kv_cache_multihead(embed_dim=8, num_heads=2, batch_size=2, kv_seq_len=6, dropout=0.0, device=torch_device)

    x1 = torch.randn(batch, 2, 8, device=torch_device)
    y1 = layer(x1, start_pos=0, rope=False)
    x2 = torch.randn(batch, 1, 8, device=torch_device)
    y2 = layer(x2, start_pos=2, rope=False)

    _checkpoint("Asserting KV cache multihead outputs and cache shape", y1_shape=y1.shape, y2_shape=y2.shape)
    assert y1.shape == x1.shape
    assert y2.shape == x2.shape
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert torch.abs(y2).mean() < 100

    k_check, _ = layer.cache.peek_kv(0, 3)
    assert k_check[:batch].shape == (batch, 2, 3, 4)


def test_kv_cache_group_query_updates_cache_and_stays_stable(torch_device):
    _checkpoint("test_kv_cache_group_query_updates_cache_and_stays_stable setup", device=torch_device)
    layer = kv_cache_group_query(embed_dim=8, num_query_heads=2, num_kv_heads=1, batch_size=2, kv_seq_len=6, dropout=0.0, device=torch_device)

    x1 = torch.randn(2, 2, 8, device=torch_device)
    y1 = layer(x1, start_pos=0, rope=False)
    x2 = torch.randn(2, 1, 8, device=torch_device)
    y2 = layer(x2, start_pos=2, rope=False)

    _checkpoint("Asserting KV cache group query outputs and cache shape", y1_shape=y1.shape, y2_shape=y2.shape)
    assert y1.shape == x1.shape
    assert y2.shape == x2.shape
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert torch.abs(y2).mean() < 100

    k_check, _ = layer.cache.peek_kv(0, 3)
    assert k_check.shape == (2, 1, 3, 4)