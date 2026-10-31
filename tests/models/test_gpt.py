import pytest
import torch
from tests._test_utils import _checkpoint
from stackformer import GPT_1, GPT_2


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
@pytest.mark.parametrize("batch,seq,embed_dim,heads", [(1, 4, 8, 1), (2, 6, 16, 4)])
def test_gpt_forward_shapes_gradients_and_finiteness(model_cls, batch, seq, embed_dim, heads, torch_device):
    _checkpoint("test_gpt_forward_shapes_gradients_and_finiteness setup", model=model_cls.__name__, batch=batch, seq=seq, device=torch_device)
    torch.manual_seed(0)
    vocab_size = 32
    x = torch.randint(0, vocab_size, (batch, seq), device=torch_device)

    model = model_cls(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=embed_dim,
        num_heads=heads,
        seq_len=seq,
        dropout=0.0,
        hidden_dim=embed_dim * 2,
        device=torch_device,
    )

    _checkpoint("Executing GPT forward pass")
    logits = model(x)
    _checkpoint("Asserting GPT logits shape and finiteness", logits_shape=logits.shape)
    assert logits.shape == (batch, seq, vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.abs(logits).mean() < 100

    loss = logits.mean()
    loss.backward()
    _checkpoint("Checking parameter gradients")
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
def test_gpt_generate_handles_single_and_batch_inputs(model_cls, torch_device):
    _checkpoint("test_gpt_generate_handles_single_and_batch_inputs setup", model=model_cls.__name__, device=torch_device)
    torch.manual_seed(0)
    vocab_size, seq_len = 32, 6
    x = torch.randint(0, vocab_size, (2, seq_len), device=torch_device)

    model = model_cls(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=8,
        num_heads=2,
        seq_len=seq_len,
        dropout=0.0,
        hidden_dim=16,
        device=torch_device,
    )

    _checkpoint("Executing GPT generate calls")
    single = model.generate(x[0], max_context_len=seq_len, max_new_tokens=2)
    batch = model.generate(x, max_context_len=seq_len, max_new_tokens=3)
    clipped = model.generate(x, max_context_len=3, max_new_tokens=2)

    _checkpoint("Asserting generated sequence shapes", single_shape=single.shape, batch_shape=batch.shape, clipped_shape=clipped.shape)
    assert single.shape == (1, seq_len + 2)
    assert batch.shape == (2, seq_len + 3)
    assert clipped.shape == (2, seq_len + 2)


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
def test_gpt_minimal_training_loop_updates_parameters(model_cls, torch_device):
    _checkpoint("test_gpt_minimal_training_loop_updates_parameters setup", model=model_cls.__name__, device=torch_device)
    torch.manual_seed(0)
    vocab_size, batch, seq_len = 24, 2, 5
    x = torch.randint(0, vocab_size, (batch, seq_len), device=torch_device)

    model = model_cls(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=8,
        num_heads=2,
        seq_len=seq_len,
        dropout=0.0,
        hidden_dim=16,
        device=torch_device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = model.lm_head.weight.detach().clone()
    _checkpoint("Running 3 optimization steps")
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        optimizer.step()
        _checkpoint("Step completed", step=step, loss=loss.item())

    _checkpoint("Asserting model parameters were updated")
    assert not torch.allclose(before, model.lm_head.weight.detach())


def test_gpt_invalid_head_configuration_raises(torch_device):
    _checkpoint("test_gpt_invalid_head_configuration_raises setup")
    with pytest.raises((AssertionError, ValueError)):
        GPT_1(
            vocab_size=20,
            num_layers=1,
            embed_dim=10,
            num_heads=3,
            seq_len=6,
            dropout=0.0,
            hidden_dim=20,
            device=torch_device,
        )
