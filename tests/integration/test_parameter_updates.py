import torch

from stackformer import GPT_1
from tests._test_utils import _checkpoint

BATCH = 2
SEQ = 6
EMB = 8
HEADS = 2
VOCAB = 32


def test_parameter_update_occurs_after_optimizer_step(torch_device):
    _checkpoint("test_parameter_update_occurs_after_optimizer_step setup", device=torch_device)
    torch.manual_seed(0)

    model = GPT_1(
        vocab_size=VOCAB,
        num_layers=1,
        embed_dim=EMB,
        num_heads=HEADS,
        seq_len=SEQ,
        hidden_dim=16,
        dropout=0.0,
        device=torch_device,
    )
    x = torch.randint(0, VOCAB, (BATCH, SEQ), device=torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    old_weight = model.embedding.weight.detach().clone()

    _checkpoint("Executing forward, backward, and optimizer step")
    logits = model(x)
    assert torch.isfinite(logits).all()
    loss = logits.mean()
    loss.backward()
    optimizer.step()

    _checkpoint("Asserting embedding weights were updated")
    assert not torch.allclose(old_weight, model.embedding.weight.detach())
