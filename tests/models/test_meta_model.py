import pytest
import torch
from tests._test_utils import _checkpoint
from stackformer import llama_1
from stackformer.models.llama import llama_2


def test_llama_1_forward_grad_and_generate(torch_device):
    _checkpoint("test_llama_1_forward_grad_and_generate setup", device=torch_device)
    torch.manual_seed(0)
    vocab_size, batch, seq_len = 32, 2, 6
    x = torch.randint(0, vocab_size, (batch, seq_len), device=torch_device)

    model = llama_1(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=8,
        num_heads=2,
        seq_len=seq_len,
        dropout=0.0,
        hidden_dim=16,
        device=torch_device,
    )
    _checkpoint("Executing llama_1 forward pass")
    logits = model(x)
    _checkpoint("Asserting llama_1 logits shape", logits_shape=logits.shape)
    assert logits.shape == (batch, seq_len, vocab_size)

    logits.mean().backward()
    _checkpoint("Checking llama_1 parameter gradients")
    assert any(p.grad is not None for p in model.parameters())

    _checkpoint("Executing llama_1 generate")
    generated = model.generate(x, max_context_len=seq_len, max_new_tokens=2)
    assert generated.shape == (batch, seq_len + 2)


def test_llama_2_forward_and_training_step(torch_device):
    _checkpoint("test_llama_2_forward_and_training_step setup", device=torch_device)
    torch.manual_seed(0)
    vocab_size, batch, seq_len = 32, 2, 6
    x = torch.randint(0, vocab_size, (batch, seq_len), device=torch_device)

    model = llama_2(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=8,
        num_query_heads=2,
        num_kv_heads=1,
        kv_seq_len=seq_len,
        batch_size=batch,
        hidden_dim=16,
        dropout=0.0,
        device=torch_device,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    _checkpoint("Executing llama_2 forward pass")
    logits = model(x)
    _checkpoint("Asserting llama_2 logits shape", logits_shape=logits.shape)
    assert logits.shape == (batch, seq_len, vocab_size)
    loss = logits.square().mean()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(logits).all()
    assert torch.abs(logits).mean() < 100


def test_llama_2_invalid_head_setup_raises(torch_device):
    _checkpoint("test_llama_2_invalid_head_setup_raises setup")
    with pytest.raises((AssertionError, ValueError)):
        llama_2(
            vocab_size=20,
            num_layers=1,
            embed_dim=8,
            num_query_heads=4,
            num_kv_heads=3,
            kv_seq_len=4,
            batch_size=1,
            hidden_dim=16,
            dropout=0.0,
            device=torch_device,
        )
