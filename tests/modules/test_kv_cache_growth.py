import pytest
import torch
from tests._test_utils import _checkpoint
from stackformer.modules.Attention import kv_cache_group_query, kv_cache_multihead

BATCH = 2
SEQ = 6
EMB = 8
HEADS = 2


@pytest.mark.parametrize("cache_cls,kwargs,kv_heads", [
    (kv_cache_multihead, dict(embed_dim=EMB, num_heads=HEADS, batch_size=BATCH, kv_seq_len=SEQ, dropout=0.0), HEADS),
    (
        kv_cache_group_query,
        dict(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=1, batch_size=BATCH, kv_seq_len=SEQ, dropout=0.0),
        1,
    ),
])
def test_kv_cache_growth_per_step_and_no_overwrite(cache_cls, kwargs, kv_heads, torch_device):
    _checkpoint("test_kv_cache_growth_per_step_and_no_overwrite setup", cache_cls=cache_cls.__name__, device=torch_device)
    x = torch.randn(BATCH, SEQ, EMB, device=torch_device)
    cache = cache_cls(device=torch_device, **kwargs)

    first_written = None
    for step in range(SEQ):
        token = x[:, step : step + 1]
        out = cache(token, start_pos=step, rope=False)
        _checkpoint("Step execution", step=step, out_shape=out.shape)
        assert out.shape == (BATCH, 1, EMB)
        assert torch.isfinite(out).all()
        
        current = cache.cache.peek_kv(0,step+1)[0]
        assert current.shape == (BATCH, kv_heads, step + 1, EMB // HEADS)

        if step == 0:
            first_written = current.clone()
        else:
            assert torch.allclose(current[:, :, :1], first_written)
