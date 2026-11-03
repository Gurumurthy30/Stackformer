import pytest
import torch
from tests._test_utils import _checkpoint

from stackformer import GPT_1, GPT_2

BATCH = 2
SEQ = 6
EMB = 8
HEADS = 2
VOCAB = 32


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
@pytest.mark.parametrize("max_new_tokens", [1, 3])
def test_generation_length_growth_never_truncates_context(model_cls, max_new_tokens, torch_device):
    _checkpoint(
        "test_generation_length_growth setup",
        model=model_cls.__name__,
        max_new_tokens=max_new_tokens,
        device=torch_device,
    )
    torch.manual_seed(0)
    x = torch.randint(0, VOCAB, (1, SEQ), device=torch_device)

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

    _checkpoint("Executing generate call")
    out = model.generate(x[0], max_context_len=SEQ, max_new_tokens=max_new_tokens)

    _checkpoint("Asserting output length equals context length plus max_new_tokens", out_len=out.shape[-1])
    assert out.shape[-1] >= x.shape[-1]
    assert out.shape[-1] == x.shape[-1] + max_new_tokens
