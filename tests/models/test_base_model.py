import pytest
import torch
import torch.nn as nn

from stackformer.config import ModelConfig
from stackformer.models import GPT1, GPT2, BaseModel, BERT, Llama1, Llama2
from stackformer.vision import ViT


def test_base_model_instantiation_with_config():
    config = ModelConfig(
        vocab_size=100,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        seq_len=16,
        hidden_dim=64,
        dropout=0.0,
    )
    model = GPT2(config)
    assert model.config is config
    assert isinstance(model, BaseModel)
    assert model.parameter_count() > 0


def test_base_model_parameter_count():
    config = ModelConfig(
        vocab_size=50,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        seq_len=8,
        hidden_dim=32,
        dropout=0.0,
    )
    model = GPT1(config)
    total = model.parameter_count()
    assert total == sum(p.numel() for p in model.parameters())

    # Freeze half of the parameters
    params = list(model.parameters())
    for p in params[: len(params) // 2]:
        p.requires_grad = False

    trainable = model.parameter_count(trainable_only=True)
    assert trainable < total
    assert trainable == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_base_model_meta_device_materialization():
    config = ModelConfig(
        vocab_size=100,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        seq_len=16,
        hidden_dim=64,
        dropout=0.0,
    )

    with torch.device("meta"):
        model = GPT2(config)

    assert any(p.is_meta for p in model.parameters())

    model.materialize(device="cpu")
    assert not any(p.is_meta for p in model.parameters())

    model.create_weights()
    x = torch.randint(0, 100, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, 100)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "method",
    ["default", "normal", "trunc_normal", "xavier_uniform", "xavier_normal", "kaiming_uniform", "orthogonal"],
)
def test_base_model_create_weights_strategies(method):
    config = ModelConfig(
        vocab_size=64,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        seq_len=8,
        hidden_dim=32,
        dropout=0.0,
    )
    model = GPT2(config)
    model.create_weights(method=method)

    x = torch.randint(0, 64, (1, 4))
    out = model(x)
    assert out.shape == (1, 4, 64)
    assert torch.isfinite(out).all()


def test_base_model_save_and_load_weights_pt(tmp_path):
    config = ModelConfig(
        vocab_size=64,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        seq_len=8,
        hidden_dim=32,
        dropout=0.0,
    )
    model = GPT1(config)
    model.eval()

    x = torch.randint(0, 64, (2, 6))
    before = model(x)

    pt_path = tmp_path / "checkpoint.pt"
    model.save_weights(pt_path)
    assert pt_path.exists()

    new_model = GPT1(config)
    new_model.eval()
    new_model.load_weights(pt_path)

    after = new_model(x)
    assert torch.allclose(before, after)


def test_base_model_save_and_load_weights_safetensors(tmp_path):
    config = ModelConfig(
        vocab_size=64,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        seq_len=8,
        hidden_dim=32,
        dropout=0.0,
    )
    model = GPT2(config)
    model.eval()

    x = torch.randint(0, 64, (2, 6))
    before = model(x)

    st_path = tmp_path / "checkpoint.safetensors"
    model.save_weights(st_path)
    assert st_path.exists()

    new_model = GPT2(config)
    new_model.eval()
    new_model.load_weights(st_path)

    after = new_model(x)
    assert torch.allclose(before, after)


def test_base_model_load_pretrained_stub():
    config = ModelConfig(
        vocab_size=64,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        seq_len=8,
        hidden_dim=32,
        dropout=0.0,
    )
    model = GPT1(config)
    # Calling load_weights() with no path triggers pretrained download stub (pass)
    result = model.load_weights()
    assert result is model


def test_vit_base_model_lifecycle(tmp_path):
    model = ViT(
        img_size=32,
        patch_size=8,
        num_layers=1,
        embed_dim=32,
        num_heads=4,
        hidden_dim=64,
        num_classes=10,
        dropout=0.0,
    )
    assert isinstance(model, BaseModel)
    assert model.parameter_count() > 0

    x = torch.randn(2, 3, 32, 32)
    before = model(x)

    st_path = tmp_path / "vit.safetensors"
    model.save_weights(st_path)
    assert st_path.exists()

    new_model = ViT(
        img_size=32,
        patch_size=8,
        num_layers=1,
        embed_dim=32,
        num_heads=4,
        hidden_dim=64,
        num_classes=10,
        dropout=0.0,
    )
    new_model.load_weights(st_path)
    after = new_model(x)
    assert torch.allclose(before, after)
