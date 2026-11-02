"""Tests for BaseModel sharding protocol, block groups, tied params, and model training."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from stackformer.models import (
    BERT,
    GPT1,
    GPT2,
    Gemma1_2B,
    Gemma1_7B,
    Llama1,
    Llama2,
    RoBERTa,
    Transformer,
)
from stackformer.models.base import BaseModel
from stackformer.training import TrainConfig, Trainer
from stackformer.vision import SegFormer, ViT


# =====================================================================
# 1. Sharding Blocks Structure and Parameter Coverage
# =====================================================================

@pytest.mark.parametrize(
    "model_fn",
    [
        lambda: GPT1(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16),
        lambda: GPT2(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16),
        lambda: Llama1(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16),
        lambda: Llama2(vocab_size=100, embed_dim=32, num_layers=2, num_query_heads=2, num_kv_heads=1, seq_len=16, hidden_dim=64),
        lambda: Gemma1_2B(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16),
        lambda: Gemma1_7B(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16),
        lambda: BERT(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16),
        lambda: RoBERTa(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16),
        lambda: Transformer(vocab_size=100, embed_dim=32, encoder_layers=2, decoder_layers=2, num_heads=2, seq_len=16),
        lambda: ViT(img_size=16, patch_size=4, in_channels=3, num_classes=5, embed_dim=32, num_layers=2, num_heads=2, hidden_dim=64),
        lambda: SegFormer(variant="b0", num_labels=5),
    ],
)
def test_model_family_sharding_blocks(model_fn):
    """Verify get_sharding_blocks returns a flat, non-empty, unique list of blocks covering >60% of params."""
    model = model_fn()
    blocks = model.get_sharding_blocks()

    assert isinstance(blocks, list), "get_sharding_blocks() must always return a flat list[nn.Module]"
    assert len(blocks) > 0, "Sharding blocks must be non-empty"
    assert len(blocks) == len(set(blocks)), "Sharding blocks must be unique"

    if isinstance(model, Transformer):
        assert len(blocks) == 4, "Transformer must return 2 encoder blocks followed by 2 decoder blocks"
    elif isinstance(model, SegFormer):
        assert len(blocks) == 4, "SegFormer must return its 4 hierarchical stages"

    for b in blocks:
        assert isinstance(b, nn.Module)

    total_params = sum(p.numel() for p in model.parameters())
    block_params = sum(p.numel() for b in blocks for p in b.parameters())
    param_fraction = block_params / total_params

    assert param_fraction > 0.60, f"Blocks cover only {param_fraction:.1%} of parameters (expected > 60%)"


def test_model_family_block_groups():
    """Verify get_block_groups returns named groups for all models."""
    # Transformer returns separate encoder and decoder
    trans = Transformer(vocab_size=100, embed_dim=32, encoder_layers=2, decoder_layers=3, num_heads=2, seq_len=16)
    trans_groups = trans.get_block_groups()
    assert set(trans_groups.keys()) == {"encoder", "decoder"}
    assert len(trans_groups["encoder"]) == 2
    assert len(trans_groups["decoder"]) == 3
    assert trans_groups["encoder"] + trans_groups["decoder"] == trans.get_sharding_blocks()

    # SegFormer returns stages
    seg = SegFormer(variant="b0", num_labels=5)
    seg_groups = seg.get_block_groups()
    assert set(seg_groups.keys()) == {"stages"}
    assert len(seg_groups["stages"]) == 4
    assert seg_groups["stages"] == seg.get_sharding_blocks()

    # Standard model returns blocks
    gpt = GPT2(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    gpt_groups = gpt.get_block_groups()
    assert set(gpt_groups.keys()) == {"blocks"}
    assert gpt_groups["blocks"] == gpt.get_sharding_blocks()


def test_model_family_tied_params():
    """Verify get_tied_params accurately reports parameter weight tying across all architectures."""
    bert = BERT(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    roberta = RoBERTa(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    gpt = GPT2(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    llama = Llama2(vocab_size=100, embed_dim=32, num_layers=2, num_query_heads=2, num_kv_heads=1, seq_len=16, hidden_dim=64)
    trans = Transformer(vocab_size=100, embed_dim=32, encoder_layers=2, decoder_layers=2, num_heads=2, seq_len=16)

    assert bert.get_tied_params() == [("token_embedding.weight", "lm_head.weight")]
    assert roberta.get_tied_params() == [("token_embedding.weight", "lm_head.weight")]
    assert gpt.get_tied_params() is None
    assert llama.get_tied_params() is None
    assert trans.get_tied_params() is None


def test_base_model_auto_detection_and_fallback():
    """Verify BaseModel auto-detects repeating layers and raises clear error when missing."""
    # 1. Successful auto-detection on custom repeating model
    class CustomTransformer(BaseModel):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50, 16)
            self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(3)])
            self.head = nn.Linear(16, 10)

    custom_model = CustomTransformer()
    blocks = custom_model.get_sharding_blocks()
    assert len(blocks) == 3
    assert blocks == list(custom_model.layers)
    assert custom_model.get_block_groups() == {"blocks": list(custom_model.layers)}
    assert custom_model.get_tied_params() is None

    # 2. Failure fallback raises informative ValueError
    class MonolithicModel(BaseModel):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)

    mono = MonolithicModel()
    with pytest.raises(ValueError, match="Could not auto-detect repeating sharding blocks"):
        mono.get_sharding_blocks()


def test_tp_plan_and_pipeline_stages_defaults():
    """Verify default implementations return None on BaseModel."""
    gpt = GPT2(vocab_size=100, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    assert gpt.get_tp_plan() is None
    assert gpt.get_pipeline_stages() is None


# =====================================================================
# 2. Five-Step Training Loss Decrease Tests for Every Model Family
# =====================================================================

def test_trainer_loss_decreases_gpt(tmp_path):
    torch.manual_seed(42)
    model = GPT2(vocab_size=64, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] * 16)
    loader = DataLoader(TensorDataset(tokens, tokens), batch_size=4, shuffle=True)

    cfg = TrainConfig(max_steps=5, lr=3e-3, optimizer="adamw", scheduler="none", checkpoint_dir=str(tmp_path), save_last=False)
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)
    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]
    assert final_loss < init_loss, f"GPT2 loss did not decrease: {init_loss} -> {final_loss}"


def test_trainer_loss_decreases_llama(tmp_path):
    torch.manual_seed(42)
    model = Llama2(vocab_size=64, embed_dim=32, num_layers=2, num_query_heads=2, num_kv_heads=1, seq_len=16, hidden_dim=64)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] * 16)
    loader = DataLoader(TensorDataset(tokens, tokens), batch_size=4, shuffle=True)

    cfg = TrainConfig(max_steps=5, lr=3e-3, optimizer="adamw", scheduler="none", checkpoint_dir=str(tmp_path), save_last=False)
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)
    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]
    assert final_loss < init_loss, f"Llama2 loss did not decrease: {init_loss} -> {final_loss}"


def test_trainer_loss_decreases_gemma(tmp_path):
    torch.manual_seed(42)
    model = Gemma1_2B(vocab_size=64, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] * 16)
    loader = DataLoader(TensorDataset(tokens, tokens), batch_size=4, shuffle=True)

    cfg = TrainConfig(max_steps=5, lr=3e-3, optimizer="adamw", scheduler="none", checkpoint_dir=str(tmp_path), save_last=False)
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)
    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]
    assert final_loss < init_loss, f"Gemma loss did not decrease: {init_loss} -> {final_loss}"


def test_trainer_loss_decreases_bert(tmp_path):
    torch.manual_seed(42)
    model = BERT(vocab_size=64, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] * 16)
    loader = DataLoader(TensorDataset(tokens, tokens), batch_size=4, shuffle=True)

    cfg = TrainConfig(max_steps=5, lr=3e-3, optimizer="adamw", scheduler="none", checkpoint_dir=str(tmp_path), save_last=False)
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)
    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]
    assert final_loss < init_loss, f"BERT loss did not decrease: {init_loss} -> {final_loss}"


def test_trainer_loss_decreases_roberta(tmp_path):
    torch.manual_seed(42)
    model = RoBERTa(vocab_size=64, embed_dim=32, num_layers=2, num_heads=2, seq_len=16)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] * 16)
    loader = DataLoader(TensorDataset(tokens, tokens), batch_size=4, shuffle=True)

    cfg = TrainConfig(max_steps=5, lr=3e-3, optimizer="adamw", scheduler="none", checkpoint_dir=str(tmp_path), save_last=False)
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)
    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]
    assert final_loss < init_loss, f"RoBERTa loss did not decrease: {init_loss} -> {final_loss}"


class Seq2SeqDataset(Dataset):
    def __init__(self, src: torch.Tensor, tgt: torch.Tensor):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            "source": self.src[idx],
            "target": self.tgt[idx],
            "labels": self.tgt[idx],
        }


def test_trainer_loss_decreases_transformer(tmp_path):
    torch.manual_seed(42)
    model = Transformer(vocab_size=32, embed_dim=32, encoder_layers=2, decoder_layers=2, num_heads=2, seq_len=8)
    src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]] * 16)
    tgt = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]] * 16)
    loader = DataLoader(Seq2SeqDataset(src, tgt), batch_size=4, shuffle=True)

    cfg = TrainConfig(max_steps=5, lr=3e-3, optimizer="adamw", scheduler="none", checkpoint_dir=str(tmp_path), save_last=False)
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)
    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]
    assert final_loss < init_loss, f"Transformer seq2seq loss did not decrease: {init_loss} -> {final_loss}"


def test_trainer_loss_decreases_vit(tmp_path):
    torch.manual_seed(42)
    model = ViT(img_size=16, patch_size=4, in_channels=3, num_classes=2, embed_dim=32, num_layers=2, num_heads=2, hidden_dim=64)
    images = torch.randn(16, 3, 16, 16)
    labels = (images[:, 0, 0, 0] > 0).long()
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=True)

    cfg = TrainConfig(max_steps=5, lr=3e-3, optimizer="adamw", loss="cross_entropy", scheduler="none", checkpoint_dir=str(tmp_path), save_last=False)
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)
    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]
    assert final_loss < init_loss, f"ViT classification loss did not decrease: {init_loss} -> {final_loss}"


def test_trainer_loss_decreases_segformer(tmp_path):
    torch.manual_seed(42)
    model = SegFormer(variant="b0", num_labels=2)
    img = torch.randn(1, 3, 32, 32)
    mask = torch.randint(0, 2, (1, 8, 8))
    images = img.repeat(20, 1, 1, 1)
    masks = mask.repeat(20, 1, 1)
    loader = DataLoader(TensorDataset(images, masks), batch_size=4, shuffle=True)

    cfg = TrainConfig(max_steps=5, lr=2e-3, optimizer="adamw", loss="cross_entropy", scheduler="none", checkpoint_dir=str(tmp_path), save_last=False)
    trainer = Trainer(model, cfg, train_loader=loader, val_loader=loader)
    init_loss = trainer.evaluate()["val_loss"]
    trainer.train()
    final_loss = trainer.evaluate()["val_loss"]
    assert final_loss < init_loss, f"SegFormer segmentation loss did not decrease: {init_loss} -> {final_loss}"
