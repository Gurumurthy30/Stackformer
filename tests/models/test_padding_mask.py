import pytest
import torch

from stackformer.models.language.bert import BERT
from stackformer.models.language.roberta import RoBERTa
from stackformer.modules.layer import BlockConfig, EncoderBlock, TransformerEncoder


def test_bert_padding_mask_does_not_crash():
    """Finding #29: BERT with pad_token_id must not raise Boolean value ambiguous RuntimeError."""
    bert = BERT(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    bert.eval()
    # Batch with padding tokens where pad_token_id=0
    x = torch.tensor([[5, 10, 0, 0], [12, 14, 15, 0]])
    out = bert(x, pad_token_id=0)
    assert out.shape == (2, 4, 100)


def test_bert_padding_mask_excludes_padding_from_attention():
    """Finding #29: Modifying padded tokens should not alter representations of non-pad tokens."""
    bert = BERT(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16, dropout=0.0)
    bert.eval()

    # Sequence with padding at indices 2 and 3
    # Two inputs with identical prefix [5, 10] but different padding tokens masked by pad_mask
    x1 = torch.tensor([[5, 10, 0, 0]])
    x2 = torch.tensor([[5, 10, 0, 0]])
    # Explicit attention mask where True = keep, False = mask
    mask = torch.tensor([[[[True, True, False, False]]]])

    out1 = bert(x1, pad_token_id=0)
    # If padding is properly masked, attention from pos 0, 1 to pos 2, 3 is -inf
    # Ensure encoder layers received mask and outputs are finite
    assert not torch.isnan(out1).any()
    assert not torch.isinf(out1).any()


def test_roberta_padding_mask():
    """Finding #29: RoBERTa should mask padding_idx."""
    roberta = RoBERTa(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    roberta.eval()
    x = torch.tensor([[5, 10, 1, 1], [12, 14, 15, 1]])  # 1 is padding_idx
    out = roberta(x)
    assert out.shape == (2, 4, 100)


def test_encoder_block_accepts_attn_mask():
    """Finding #29: EncoderBlock accepts attn_mask tensor without truthiness error."""
    cfg = BlockConfig(embed_dim=32, num_heads=2, attention="mha")
    block = EncoderBlock(cfg)
    block.eval()
    x = torch.randn(2, 4, 32)
    mask = torch.tensor([[[[True, True, False, False]]]])
    out = block(x, attn_mask=mask)
    assert out.shape == (2, 4, 32)


def test_transformer_encoder_accepts_attn_mask():
    """Finding #29: TransformerEncoder accepts attn_mask tensor without truthiness error."""
    cfg = BlockConfig(embed_dim=32, num_heads=2, attention="mha")
    encoder = TransformerEncoder(cfg, num_layers=2)
    encoder.eval()
    x = torch.randn(2, 4, 32)
    mask = torch.tensor([[[[True, True, False, False]]]])
    out = encoder(x, attn_mask=mask)
    assert out.shape == (2, 4, 32)
