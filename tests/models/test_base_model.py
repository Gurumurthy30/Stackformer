import pytest
import torch
import torch.nn as nn

from stackformer.config import ModelConfig
from stackformer.models import (
    BERT,
    GPT1,
    GPT2,
    BaseModel,
    Gemma1_2B,
    Gemma1_7B,
    Llama1,
    Llama2,
    RoBERTa,
    Transformer,
)
from stackformer.vision import SegFormer, ViT


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
    [
        "default",
        "normal",
        "trunc_normal",
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "orthogonal",
    ],
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
    from stackformer.training import TrainConfig, Trainer

    trainer = Trainer(model=model, config=TrainConfig(checkpoint_dir=str(tmp_path)))
    trainer.save(pt_path)
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
    from stackformer.training import TrainConfig, Trainer

    trainer = Trainer(model=model, config=TrainConfig(checkpoint_dir=str(tmp_path)))
    trainer.save(st_path)
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
    from stackformer.training import TrainConfig, Trainer

    trainer = Trainer(model=model, config=TrainConfig(checkpoint_dir=str(tmp_path)))
    trainer.save(st_path)
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


def test_base_model_has_no_save_weights():
    """Verify single save path contract: models have no save_weights attribute."""
    config = ModelConfig(
        vocab_size=32, embed_dim=16, num_layers=1, num_heads=2, seq_len=8, hidden_dim=32
    )
    model = GPT2(config)
    assert not hasattr(model, "save_weights"), "Model must not have save_weights method"
    assert not hasattr(BaseModel, "save_weights"), "BaseModel must not have save_weights method"


def test_standalone_load_weights_and_dir_loading(tmp_path):
    """Test stackformer.load_weights loads from file and directory checkpoints."""
    import stackformer
    from stackformer.training import TrainConfig, Trainer

    config = ModelConfig(
        vocab_size=64, embed_dim=16, num_layers=1, num_heads=2, seq_len=8, hidden_dim=32
    )
    model = GPT2(config)
    model.eval()

    x = torch.randint(0, 64, (2, 4))
    expected = model(x)

    # 1. Save via trainer.save()
    file_path = tmp_path / "model.safetensors"
    trainer = Trainer(model=model, config=TrainConfig(checkpoint_dir=str(tmp_path)))
    trainer.save(file_path)

    fresh_model1 = GPT2(config)
    fresh_model1.eval()
    stackformer.load_weights(fresh_model1, file_path)
    assert torch.allclose(fresh_model1(x), expected)

    # 2. Save via trainer.save_checkpoint() to directory
    ckpt_path = trainer.save_checkpoint()
    fresh_model2 = GPT2(config)
    fresh_model2.eval()
    stackformer.load_weights(fresh_model2, ckpt_path)
    assert torch.allclose(fresh_model2(x), expected)


def test_load_weights_prefix_stripping(tmp_path):
    """Test loading weights with module. and _orig_mod. prefixes strips cleanly."""
    import stackformer

    config = ModelConfig(
        vocab_size=64, embed_dim=16, num_layers=1, num_heads=2, seq_len=8, hidden_dim=32
    )
    model = GPT2(config)
    model.eval()

    raw_state = model.state_dict()
    prefixed_state = {}
    for k, v in raw_state.items():
        prefixed_state[f"module._orig_mod.{k}"] = v

    pt_path = tmp_path / "prefixed.pt"
    torch.save(prefixed_state, pt_path)

    fresh = GPT2(config)
    fresh.eval()
    stackformer.load_weights(fresh, pt_path)

    x = torch.randint(0, 64, (2, 4))
    assert torch.allclose(fresh(x), model(x))


def test_load_weights_error_handling(tmp_path):
    """Test load_weights error paths: missing file, empty dir, strict key mismatch."""
    import stackformer

    config = ModelConfig(
        vocab_size=64, embed_dim=16, num_layers=1, num_heads=2, seq_len=8, hidden_dim=32
    )
    model = GPT2(config)

    # Non-existent file
    with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
        stackformer.load_weights(model, tmp_path / "non_existent.safetensors")

    # Empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No model.safetensors or model.pt found"):
        stackformer.load_weights(model, empty_dir)

    # Strict key mismatch
    diff_config = ModelConfig(
        vocab_size=128, embed_dim=32, num_layers=2, num_heads=4, seq_len=8, hidden_dim=64
    )
    other_model = GPT2(diff_config)
    other_file = tmp_path / "other.pt"
    torch.save(other_model.state_dict(), other_file)

    with pytest.raises(RuntimeError):
        stackformer.load_weights(model, other_file, strict=True)


def test_all_model_families_materialize_and_finite_parameters():
    cfg = ModelConfig(
        vocab_size=100,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        seq_len=16,
        hidden_dim=64,
        dropout=0.0,
    )

    models = [
        ("GPT1", lambda: GPT1(cfg)),
        ("GPT2", lambda: GPT2(cfg)),
        ("BERT", lambda: BERT(cfg)),
        ("RoBERTa", lambda: RoBERTa(cfg)),
        ("Llama1", lambda: Llama1(cfg)),
        (
            "Llama2",
            lambda: Llama2(
                vocab_size=100,
                embed_dim=32,
                num_layers=2,
                num_query_heads=4,
                num_kv_heads=2,
                seq_len=16,
                hidden_dim=64,
            ),
        ),
        ("Gemma1_2B", lambda: Gemma1_2B(cfg)),
        ("Gemma1_7B", lambda: Gemma1_7B(cfg)),
        ("Transformer", lambda: Transformer(cfg)),
        (
            "ViT",
            lambda: ViT(
                img_size=32,
                patch_size=8,
                num_layers=2,
                embed_dim=32,
                num_heads=4,
                hidden_dim=64,
                num_classes=10,
            ),
        ),
        ("SegFormer", lambda: SegFormer(variant="b0", num_labels=10)),
    ]

    for name, init_fn in models:
        with torch.device("meta"):
            model = init_fn()
        model.materialize("cpu")
        model.create_weights("default")
        for p_name, p in model.named_parameters():
            assert torch.isfinite(p).all(), (
                f"{name} parameter {p_name} is not finite after materialize and create_weights"
            )
        for b_name, b in model.named_buffers():
            assert torch.isfinite(b).all(), (
                f"{name} buffer {b_name} is not finite after materialize and create_weights"
            )
