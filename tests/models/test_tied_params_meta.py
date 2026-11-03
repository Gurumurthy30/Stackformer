import pytest
import torch

from stackformer.models import BERT, GPT2, Llama2, RoBERTa


def test_get_tied_params_meta_tensors_skipped():
    """Verify get_tied_params() skips meta tensors and does not fail on unallocated memory."""
    with torch.device("meta"):
        model = GPT2(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2)

    # Must not raise RuntimeError on meta data_ptr
    tied = model.get_tied_params()
    assert tied is None

    # Materialize and verify fallback still returns None when untied
    model.materialize()
    assert model.get_tied_params() is None

    # Dynamically tie parameters and verify data_ptr detection
    model.lm_head.weight = model.embedding.weight
    tied_after = model.get_tied_params()
    assert tied_after is not None
    assert len(tied_after) == 1
    assert set(tied_after[0]) == {"embedding.weight", "lm_head.weight"}


def test_get_tied_params_bert_roberta_meta():
    """BERT and RoBERTa report tied parameters even on meta device."""
    with torch.device("meta"):
        bert = BERT(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2)
        roberta = RoBERTa(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2)

    assert bert.get_tied_params() == [("token_embedding.weight", "lm_head.weight")]
    assert roberta.get_tied_params() == [("token_embedding.weight", "lm_head.weight")]
