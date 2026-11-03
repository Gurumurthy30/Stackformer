import pytest
import torch

from stackformer import GPT_2, gemma_1_2b, llama_1
from tests._test_utils import _checkpoint

BATCH = 2
SEQ = 6
EMB = 8
HEADS = 2
VOCAB = 32


@pytest.mark.parametrize("model_cls", [GPT_2, gemma_1_2b, llama_1])
def test_generation_numerical_stability_single_and_batch(model_cls, torch_device):
    _checkpoint("test_generation_numerical_stability setup", model=model_cls.__name__, device=torch_device)
    torch.manual_seed(0)
    x = torch.randint(0, VOCAB, (BATCH, SEQ), device=torch_device)

    model = model_cls(
        vocab_size=VOCAB,
        num_layers=1,
        embed_dim=EMB,
        num_heads=HEADS,
        seq_len=SEQ,
        hidden_dim=16,
        dropout=0.0,
        device=torch_device,
    )

    _checkpoint("Running forward pass logits check")
    logits = model(x)
    assert torch.isfinite(logits).all()

    _checkpoint("Running single and batch generation for numerical stability")
    single = model.generate(x[0], max_context_len=SEQ, max_new_tokens=2)
    batch = model.generate(x, max_context_len=SEQ, max_new_tokens=2)

    _checkpoint("Asserting finiteness of generated tokens", single_shape=single.shape, batch_shape=batch.shape)
    assert single.shape == (1, SEQ + 2)
    assert batch.shape == (BATCH, SEQ + 2)
    assert torch.isfinite(single.float()).all()
    assert torch.isfinite(batch.float()).all()
