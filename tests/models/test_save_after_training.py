import pytest
import torch

from stackformer import GPT_1, GPT_2
from tests._test_utils import _checkpoint


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
def test_save_load_after_training_keeps_predictions(tmp_path, model_cls, torch_device):
    _checkpoint(
        "test_save_load_after_training_keeps_predictions setup",
        model=model_cls.__name__,
        device=torch_device,
    )
    torch.manual_seed(0)
    x = torch.randint(0, 32, (2, 6), device=torch_device)

    model = model_cls(
        vocab_size=32,
        num_layers=1,
        embed_dim=8,
        num_heads=2,
        seq_len=6,
        dropout=0.0,
        hidden_dim=16,
        device=torch_device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    _checkpoint("Training model for 3 iterations")
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        optimizer.step()

    model.eval()
    expected = model(x)

    path = tmp_path / "trained.pt"
    _checkpoint("Saving model state_dict", path=str(path))
    torch.save(model.state_dict(), path)

    reloaded = model_cls(
        vocab_size=32,
        num_layers=1,
        embed_dim=8,
        num_heads=2,
        seq_len=6,
        dropout=0.0,
        hidden_dim=16,
        device=torch_device,
    )
    reloaded.load_state_dict(torch.load(path, map_location=str(torch_device)))
    reloaded.eval()

    _checkpoint("Executing reloaded model forward pass")
    got = reloaded(x)
    _checkpoint("Asserting loaded model predictions match expected")
    assert torch.isfinite(got).all()
    assert torch.allclose(expected, got)
