import pytest
import torch
import torch.nn as nn

from stackformer.initialization import (
    Initializer,
    init_weights,
    kaiming_normal_,
    kaiming_uniform_,
    normal_init_,
    ones_init_,
    orthogonal_init_,
    trunc_normal_init_,
    uniform_init_,
    xavier_normal_,
    xavier_uniform_,
    zeros_init_,
)


def test_functional_initializers():
    t = torch.empty(10, 10)

    zeros_init_(t)
    assert (t == 0).all()

    ones_init_(t)
    assert (t == 1).all()

    normal_init_(t, mean=0.0, std=0.02)
    assert t.std() > 0

    uniform_init_(t, a=-0.05, b=0.05)
    assert (t >= -0.05).all() and (t <= 0.05).all()

    trunc_normal_init_(t, mean=0.0, std=0.02, a=-0.04, b=0.04)
    assert (t >= -0.04).all() and (t <= 0.04).all()

    xavier_uniform_(t)
    assert t.std() > 0

    xavier_normal_(t)
    assert t.std() > 0

    kaiming_uniform_(t)
    assert t.std() > 0

    kaiming_normal_(t)
    assert t.std() > 0

    orthogonal_init_(t)
    # Check orthogonality: W @ W.T ~ I
    eye = torch.matmul(t, t.T)
    assert torch.allclose(eye, torch.eye(10), atol=1e-4)


def test_initializer_class():
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.LayerNorm(64),
        nn.Embedding(100, 32, padding_idx=0),
    )

    initializer = Initializer(strategy="xavier_uniform")
    initializer.apply(model)

    # Check LayerNorm weights are 1 and biases are 0
    assert torch.allclose(model[1].weight, torch.ones_like(model[1].weight))
    assert torch.allclose(model[1].bias, torch.zeros_like(model[1].bias))

    # Check padding idx in embedding is zeroed
    assert (model[2].weight[0] == 0).all()


def test_initializer_invalid_strategy():
    with pytest.raises(ValueError, match="Unknown initialization strategy"):
        Initializer(strategy="non_existent_strategy")


def test_init_weights_helper():
    linear = nn.Linear(16, 16)
    init_weights(linear, strategy="normal", std=0.01)
    assert linear.weight.std() > 0
    assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))

