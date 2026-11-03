import pytest
import torch
import torch.nn as nn
from tests._test_utils import _checkpoint

from stackformer import GPT_1, GPT_2, gemma_1_2b, llama_1
from stackformer.generate import text_generate


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2, gemma_1_2b, llama_1])
def test_generation_shape_and_stability_single_and_batch(model_cls, torch_device):
    _checkpoint("test_generation_shape_and_stability setup", model=model_cls.__name__, device=torch_device)
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

    _checkpoint("Running single and batch generation")
    single = model.generate(x[0], max_context_len=6, max_new_tokens=3)
    batch = model.generate(x, max_context_len=6, max_new_tokens=2)

    _checkpoint("Asserting output shapes and finiteness", single_shape=single.shape, batch_shape=batch.shape)
    assert single.shape == (1, 9)
    assert batch.shape == (2, 8)
    assert torch.isfinite(single.float()).all() and torch.isfinite(batch.float()).all()


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
def test_generation_is_deterministic_with_seed_reset(model_cls, torch_device):
    _checkpoint("test_generation_is_deterministic_with_seed_reset setup", model=model_cls.__name__, device=torch_device)
    x = torch.randint(0, 32, (1, 6), device=torch_device)
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

    torch.manual_seed(0)
    out1 = model.generate(x, max_context_len=6, max_new_tokens=3)
    torch.manual_seed(0)
    out2 = model.generate(x, max_context_len=6, max_new_tokens=3)

    _checkpoint("Asserting deterministic generation outputs equal")
    assert torch.equal(out1, out2)


class _FixedLogitModel(nn.Module):
    def __init__(self, vocab_size: int = 8, eos_token_id: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self._decode_step = 0

    def prefill(self, input_ids):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab_size)
        logits[:, -1, 2] = 10.0
        return logits, {"seen": t}

    def decode(self, next_token, cache):
        self._decode_step += 1
        b = next_token.size(0)
        logits = torch.zeros(b, 1, self.vocab_size)
        logits[0, 0, self.eos_token_id] = 10.0
        logits[1, 0, self.eos_token_id if self._decode_step > 1 else 2] = 10.0
        return logits, cache

    def forward(self, input_ids):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab_size)
        logits[:, -1, self.eos_token_id] = 10.0
        return logits

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return text_generate(self, *args, **kwargs)


def test_generation_batch_eos_tracking_mixed_completion():
    _checkpoint("test_generation_batch_eos_tracking_mixed_completion setup")
    model = _FixedLogitModel()
    prompt = torch.tensor([[3, 4], [5, 6]])

    _checkpoint("Executing generate on mock EOS model")
    out = model.generate(
        prompt,
        max_context_len=8,
        max_new_tokens=4,
        eos_token_id=1,
    )

    _checkpoint("Asserting EOS tracking output shape", out_shape=out.shape)
    assert out.shape[0] == 2
    assert out.shape[1] == prompt.shape[1] + 3
