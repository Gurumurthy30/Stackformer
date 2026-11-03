import torch

from stackformer import GPT_2
from tests._test_utils import _checkpoint

BATCH = 2
SEQ = 6
EMB = 8
HEADS = 2
VOCAB = 32


def test_full_stackformer_pipeline_forward_backward_and_generation(torch_device):
    _checkpoint(
        "test_full_stackformer_pipeline_forward_backward_and_generation setup", device=torch_device
    )
    torch.manual_seed(0)

    tokens = torch.randint(0, VOCAB, (BATCH, SEQ), device=torch_device)
    model = GPT_2(
        vocab_size=VOCAB,
        num_layers=1,
        embed_dim=EMB,
        num_heads=HEADS,
        seq_len=SEQ,
        hidden_dim=16,
        dropout=0.0,
        device=torch_device,
    )

    _checkpoint("Executing forward pass on full GPT_2 stack")
    logits = model(tokens)
    _checkpoint("Asserting logits shape and finiteness", logits_shape=logits.shape)
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert torch.isfinite(logits).all()

    loss = logits.mean()
    loss.backward()
    _checkpoint("Checking backward pass gradients")
    assert any(p.grad is not None for p in model.parameters())

    _checkpoint("Executing generation call")
    generated = model.generate(tokens[0], max_context_len=SEQ, max_new_tokens=2)
    _checkpoint("Asserting generated output shape", shape=generated.shape)
    assert generated.shape == (1, SEQ + 2)
    assert torch.isfinite(generated.float()).all()
