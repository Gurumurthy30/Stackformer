import torch
from tests._test_utils import _checkpoint

from stackformer import GPT_1


def test_minimal_training_loop_runs_and_updates_weights(torch_device):
    _checkpoint("test_minimal_training_loop_runs_and_updates_weights setup", device=torch_device)
    torch.manual_seed(0)
    model = GPT_1(
        vocab_size=32,
        num_layers=1,
        embed_dim=8,
        num_heads=2,
        seq_len=6,
        dropout=0.0,
        hidden_dim=16,
        device=torch_device,
    )
    x = torch.randint(0, 32, (2, 6), device=torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = model.lm_head.weight.detach().clone()
    _checkpoint("Running training loop steps")
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        assert torch.isfinite(out).all()
        loss = out.mean()
        loss.backward()
        optimizer.step()
        _checkpoint("Step completed", step=step, loss=loss.item())

    _checkpoint("Asserting model weights updated")
    assert not torch.allclose(before, model.lm_head.weight.detach())
