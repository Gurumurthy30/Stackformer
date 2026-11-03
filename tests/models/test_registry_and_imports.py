"""Tests for model categorization, registry (list_models, get_model), and imports."""

import pytest
import torch
import torch.nn as nn


def test_category_imports():
    """Verify all models can be imported from their respective category subpackages."""
    from stackformer.models.language import (
        BERT,
        GPT1,
        GPT2,
        GPT_1,
        GPT_2,
        Decoder,
        EncoderDecoder,
        Gemma1_2B,
        Gemma1_7B,
        Gemma_1_2B,
        Gemma_1_7B,
        GPTDecoder,
        Llama1,
        Llama2,
        Llama_1,
        Llama_2,
        RoBERTa,
        Transformer,
        gemma_1_2b,
        gemma_1_7b,
        llama_1,
        llama_2,
    )
    from stackformer.models.vision import (
        SegFormer,
        SegFormerB0,
        ViT,
        segformer,
        vit,
    )

    assert issubclass(GPT2, nn.Module)
    assert issubclass(Llama2, nn.Module)
    assert issubclass(Gemma1_2B, nn.Module)
    assert issubclass(BERT, nn.Module)
    assert issubclass(RoBERTa, nn.Module)
    assert issubclass(Transformer, nn.Module)
    assert issubclass(Decoder, nn.Module)
    assert issubclass(EncoderDecoder, nn.Module)
    assert issubclass(ViT, nn.Module)
    assert issubclass(SegFormer, nn.Module)
    assert vit is ViT
    assert segformer is SegFormer


def test_flat_and_toplevel_imports():
    """Verify flat imports from stackformer.models and top-level stackformer."""
    import stackformer
    from stackformer.models import (
        BERT,
        GPT2,
        BaseModel,
        Gemma1_2B,
        Llama2,
        RoBERTa,
        SegFormer,
        Transformer,
        ViT,
        get_model,
        list_models,
    )

    assert BaseModel is not None
    assert issubclass(GPT2, BaseModel)
    assert issubclass(ViT, BaseModel)
    assert issubclass(SegFormer, BaseModel)
    assert callable(list_models)
    assert callable(get_model)

    assert hasattr(stackformer, "GPT2")
    assert hasattr(stackformer, "ViT")
    assert hasattr(stackformer, "SegFormer")


def test_list_models():
    """Verify list_models returns expected categories and architecture names."""
    from stackformer.models import list_models

    models_by_cat = list_models()
    assert isinstance(models_by_cat, dict)
    assert "language" in models_by_cat
    assert "vision" in models_by_cat

    lang = models_by_cat["language"]
    for expected in [
        "BERT",
        "GPT1",
        "GPT2",
        "Gemma1_2B",
        "Gemma1_7B",
        "Llama1",
        "Llama2",
        "RoBERTa",
        "Transformer",
    ]:
        assert expected in lang

    vis = models_by_cat["vision"]
    for expected in ["ViT", "SegFormer"]:
        assert expected in vis


def test_get_model_tiny_configs():
    """Verify get_model instantiates each model family with tiny configs."""
    from stackformer.models import get_model
    from stackformer.models.language import (
        BERT,
        GPT2,
        Gemma1_2B,
        Llama2,
        RoBERTa,
        Transformer,
    )
    from stackformer.models.vision import SegFormer, ViT

    # Language models
    m_gpt = get_model("gpt2", vocab_size=50, num_layers=2, embed_dim=32, num_heads=2, seq_len=16)
    assert isinstance(m_gpt, GPT2)

    m_llama = get_model(
        "Llama2",
        vocab_size=50,
        num_layers=2,
        embed_dim=32,
        num_query_heads=4,
        num_kv_heads=2,
        seq_len=16,
    )
    assert isinstance(m_llama, Llama2)

    m_gemma = get_model(
        "gemma1_2b", vocab_size=50, num_layers=2, embed_dim=32, num_heads=2, seq_len=16
    )
    assert isinstance(m_gemma, Gemma1_2B)

    m_bert = get_model("BERT", vocab_size=50, num_layers=2, embed_dim=32, num_heads=2, seq_len=16)
    assert isinstance(m_bert, BERT)

    m_roberta = get_model(
        "roberta", vocab_size=50, num_layers=2, embed_dim=32, num_heads=2, seq_len=16
    )
    assert isinstance(m_roberta, RoBERTa)

    m_tf = get_model(
        "Transformer",
        vocab_size=50,
        embed_dim=32,
        num_heads=2,
        hidden_dim=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_seq_len=16,
    )
    assert isinstance(m_tf, Transformer)

    # Vision models
    m_vit = get_model(
        "vit", img_size=16, patch_size=4, num_layers=2, embed_dim=32, num_heads=2, num_classes=5
    )
    assert isinstance(m_vit, ViT)

    m_seg = get_model("SegFormer", variant="B0", num_labels=5)
    assert isinstance(m_seg, SegFormer)


def test_get_model_unknown_raises():
    """Verify get_model raises ValueError on unknown architecture name."""
    from stackformer.models import get_model

    with pytest.raises(ValueError, match="Unknown model architecture 'nonexistent'"):
        get_model("nonexistent")
