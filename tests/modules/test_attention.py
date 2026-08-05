import pytest
import torch
from tests._test_utils import _checkpoint

from stackformer.modules.Attention import (
    Cross_MultiHead_Attention,
    Group_query_Attention,
    Group_query_Attention_With_RoPE,
    Multi_Head_Attention,
    Multi_Head_Attention_With_RoPE,
    Multi_query_Attention,
    Multi_query_Attention_With_RoPE,
    Self_Attention,
    kv_cache_group_query,
    kv_cache_multihead,
    _ROPE_FREQ_CACHE,
)
from stackformer.modules.attention_engine import _run_attention, _run_flex_attention, _run_sdpa

BATCH, SEQ, EMB = 2, 6, 8
HEADS, KV_HEADS = 2, 1
MASKS = [
    dict(mask_type=["causal"]),
    dict(mask_type=["sliding_window"], window_size=2),
    dict(mask_type=["sliding_window", "global_mask"], window_size=2, global_index=[1]),
    dict(mask_type=["no"]),
]


def _assert_finite_and_grad(out: torch.Tensor, x: torch.Tensor) -> None:
    _checkpoint("_assert_finite_and_grad checking shapes and finite values", out_shape=out.shape, x_shape=x.shape)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.abs(out).mean() < 100

    loss = out.square().mean()
    loss.backward()
    _checkpoint("Checking backward pass gradients")
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("mask_kw", MASKS)
@pytest.mark.parametrize(
    "attn_ctor",
    [
        lambda kw, dev: Self_Attention(EMB, dropout=0.0, device=dev, **kw),
        lambda kw, dev: Multi_Head_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=dev, **kw),
        lambda kw, dev: Multi_Head_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=dev, **kw),
        lambda kw, dev: Multi_query_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=dev, **kw),
        lambda kw, dev: Multi_query_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=dev, **kw),
        lambda kw, dev: Group_query_Attention(
            embed_dim=EMB,
            num_query_heads=HEADS,
            num_kv_heads=KV_HEADS,
            dropout=0.0,
            device=dev,
            **kw,
        ),
        lambda kw, dev: Group_query_Attention_With_RoPE(
            embed_dim=EMB,
            num_query_heads=HEADS,
            num_kv_heads=KV_HEADS,
            dropout=0.0,
            device=dev,
            **kw,
        ),
    ],
)
def test_attention_variants_masks_and_gradients(mask_kw, attn_ctor, torch_device):
    _checkpoint("test_attention_variants_masks_and_gradients setup", mask_kw=mask_kw, device=torch_device)
    x = torch.randn(BATCH, SEQ, EMB, requires_grad=True, device=torch_device)
    layer = attn_ctor(mask_kw, torch_device)
    _checkpoint("Executing forward pass on attention variant", layer_class=layer.__class__.__name__)
    out = layer(x)
    _assert_finite_and_grad(out, x)


def test_cross_attention_shape_and_gradients(torch_device):
    _checkpoint("test_cross_attention_shape_and_gradients setup", device=torch_device)
    q = torch.randn(BATCH, SEQ, EMB, requires_grad=True, device=torch_device)
    ctx = torch.randn(BATCH, SEQ, EMB, device=torch_device)
    layer = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, mask_type=["causal"], device=torch_device)

    _checkpoint("Executing Cross_MultiHead_Attention forward pass")
    out = layer(q, context=ctx)
    _assert_finite_and_grad(out, q)


@pytest.mark.parametrize("batch", [1, 2])
def test_kv_cache_multihead_multiple_decode_steps(batch, torch_device):
    _checkpoint("test_kv_cache_multihead_multiple_decode_steps setup", batch=batch, device=torch_device)
    cache = kv_cache_multihead(embed_dim=EMB, num_heads=HEADS, batch_size=2, kv_seq_len=SEQ, dropout=0.0, device=torch_device)

    x0 = torch.randn(batch, 2, EMB, device=torch_device)
    _checkpoint("KV cache step 0 forward", x0_shape=x0.shape)
    y0 = cache(x0, start_pos=0)
    assert y0.shape == x0.shape
    k_check, _ = cache.cache.peek_kv(0, 2)
    assert k_check[:batch].shape == (batch, HEADS, 2, EMB // HEADS)

    x1 = torch.randn(batch, 1, EMB, device=torch_device)
    _checkpoint("KV cache step 1 forward", x1_shape=x1.shape)
    y1 = cache(x1, start_pos=2)
    assert y1.shape == x1.shape
    assert torch.isfinite(y1).all()
    assert torch.abs(y1).mean() < 100


def test_kv_cache_group_query_multiple_decode_steps(torch_device):
    _checkpoint("test_kv_cache_group_query_multiple_decode_steps setup", device=torch_device)
    cache = kv_cache_group_query(
        embed_dim=EMB,
        num_query_heads=HEADS,
        num_kv_heads=KV_HEADS,
        batch_size=BATCH,
        kv_seq_len=SEQ,
        dropout=0.0,
        device=torch_device,
    )

    x0 = torch.randn(BATCH, 2, EMB, device=torch_device)
    _checkpoint("Group query KV cache step 0 forward", x0_shape=x0.shape)
    y0 = cache(x0, start_pos=0, rope=True)
    assert y0.shape == x0.shape
    k_check, _ = cache.cache.peek_kv(0, 2)
    assert k_check.shape == (BATCH, KV_HEADS, 2, EMB // HEADS)

    x1 = torch.randn(BATCH, 1, EMB, device=torch_device)
    _checkpoint("Group query KV cache step 1 forward", x1_shape=x1.shape)
    y1 = cache(x1, start_pos=2, rope=False)
    assert y1.shape == x1.shape
    assert torch.isfinite(y1).all()
    assert torch.abs(y1).mean() < 100


@pytest.mark.parametrize(
    "ctor,args,error",
    [
        (Multi_Head_Attention, dict(embed_dim=10, num_heads=3), AssertionError),
        (Group_query_Attention, dict(embed_dim=8, num_query_heads=3, num_kv_heads=2), AssertionError),
        (Self_Attention, dict(embed_dim=8, mask_type=["does_not_exist"]), ValueError),
    ],
)
def test_attention_invalid_configs_raise(ctor, args, error, torch_device):
    _checkpoint("test_attention_invalid_configs_raise", ctor=ctor.__name__, args=args)
    with pytest.raises(error):
        layer = ctor(device=torch_device, **args)
        if ctor is Self_Attention:
            layer(torch.randn(1, 4, 8, device=torch_device))


def test_cross_attention_mismatched_batch_raises(torch_device):
    _checkpoint("test_cross_attention_mismatched_batch_raises setup", device=torch_device)
    layer = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=torch_device)
    q = torch.randn(2, 4, EMB, device=torch_device)
    ctx = torch.randn(1, 4, EMB, device=torch_device)
    with pytest.raises(RuntimeError):
        layer(q, context=ctx)


def test_cross_attention_accepts_explicit_t_s_mask(torch_device):
    _checkpoint("test_cross_attention_accepts_explicit_t_s_mask setup", device=torch_device)
    layer = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=torch_device)
    q = torch.randn(BATCH, 4, EMB, requires_grad=True, device=torch_device)
    ctx = torch.randn(BATCH, 6, EMB, device=torch_device)
    mask = torch.zeros(4, 6, dtype=torch.bool, device=torch_device)
    out = layer(q, context=ctx, attn_mask=mask)
    _assert_finite_and_grad(out, q)


def test_cross_attention_invalid_mask_shape_raises(torch_device):
    _checkpoint("test_cross_attention_invalid_mask_shape_raises setup", device=torch_device)
    layer = Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=torch_device)
    q = torch.randn(BATCH, 4, EMB, device=torch_device)
    ctx = torch.randn(BATCH, 6, EMB, device=torch_device)
    bad_mask = torch.zeros(4, 4, dtype=torch.bool, device=torch_device)
    with pytest.raises(ValueError):
        layer(q, context=ctx, attn_mask=bad_mask)


def test_attention_constructors_allow_positional_dropout_safely(torch_device):
    _checkpoint("test_attention_constructors_allow_positional_dropout_safely setup", device=torch_device)
    x = torch.randn(1, 3, EMB, device=torch_device)
    layer = Multi_Head_Attention_With_RoPE(EMB, HEADS, 0.0, device=torch_device)
    out = layer(x)
    assert out.shape == x.shape


def test_attention_rope_freq_cache_tracking(torch_device):
    """Cover M-01 issue monitoring module-level _ROPE_FREQ_CACHE."""
    _checkpoint("test_attention_rope_freq_cache_tracking starting", initial_cache_len=len(_ROPE_FREQ_CACHE))
    layer = Multi_Head_Attention_With_RoPE(embed_dim=8, num_heads=2, device=torch_device)
    x = torch.randn(1, 4, 8, device=torch_device)
    out = layer(x)
    _checkpoint("Checking RoPE cache presence", cache_size=len(_ROPE_FREQ_CACHE))
    assert isinstance(_ROPE_FREQ_CACHE, dict)


def test_run_attention_dispatcher_defaults(torch_device):
    _checkpoint("test_run_attention_dispatcher_defaults setup", device=torch_device)
    q = torch.randn(2, 2, 4, 8, device=torch_device)
    k = torch.randn(2, 2, 4, 8, device=torch_device)
    v = torch.randn(2, 2, 4, 8, device=torch_device)

    # Calling _run_attention with default backend ("sdpa")
    out_default = _run_attention(q=q, k=k, v=v)
    out_sdpa = _run_attention(backend="sdpa", q=q, k=k, v=v)
    assert out_default.shape == (2, 2, 4, 8)
    assert torch.allclose(out_default, out_sdpa)


def test_run_attention_invalid_backend_raises(torch_device):
    _checkpoint("test_run_attention_invalid_backend_raises setup", device=torch_device)
    q = torch.randn(2, 2, 4, 8, device=torch_device)
    with pytest.raises(ValueError, match="Unknown backend"):
        _run_attention(backend="invalid_backend", q=q, k=q, v=q)


@pytest.mark.parametrize(
    "ctor",
    [
        lambda dev, backend: Self_Attention(EMB, dropout=0.0, device=dev, backend=backend),
        lambda dev, backend: Multi_Head_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=dev, backend=backend),
        lambda dev, backend: Multi_Head_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=dev, backend=backend),
        lambda dev, backend: Cross_MultiHead_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=dev, backend=backend),
        lambda dev, backend: Multi_query_Attention(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=dev, backend=backend),
        lambda dev, backend: Multi_query_Attention_With_RoPE(embed_dim=EMB, num_heads=HEADS, dropout=0.0, device=dev, backend=backend),
        lambda dev, backend: Group_query_Attention(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=KV_HEADS, dropout=0.0, device=dev, backend=backend),
        lambda dev, backend: Group_query_Attention_With_RoPE(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=KV_HEADS, dropout=0.0, device=dev, backend=backend),
        lambda dev, backend: kv_cache_multihead(embed_dim=EMB, num_heads=HEADS, batch_size=BATCH, kv_seq_len=SEQ, dropout=0.0, device=dev, backend=backend),
        lambda dev, backend: kv_cache_group_query(embed_dim=EMB, num_query_heads=HEADS, num_kv_heads=KV_HEADS, batch_size=BATCH, kv_seq_len=SEQ, dropout=0.0, device=dev, backend=backend),
    ],
)
def test_attention_modules_backend_support(ctor, torch_device):
    _checkpoint("test_attention_modules_backend_support setup", device=torch_device)
    layer_sdpa = ctor(torch_device, "sdpa")
    assert layer_sdpa.backend == "sdpa"

    layer_flex = ctor(torch_device, "flex")
    assert layer_flex.backend == "flex"

    with pytest.raises(ValueError):
        ctor(torch_device, "unsupported_backend")

