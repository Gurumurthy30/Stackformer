"""Configuration dataclasses for StackFormer model architectures, training, and generation.

Provides structured configuration containers for:
- Model architecture hyperparameter definitions (`ModelConfig`)
- Optimization, scheduling, and training loop parameters (`TrainingConfig`)
- Text generation decoding parameters (`GenerationConfig`)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    """Configuration container for Transformer model architecture hyperparameters.

    Note:
        Model constructors (such as ``Gemma1_2B`` or ``Llama1``) accept individual
        hyperparameter keyword arguments directly; ``ModelConfig`` serves as a
        structured container for storing, serializing, or passing configuration dictionaries
        via ``asdict(config)``.

    Args:
        vocab_size (int): Size of the token vocabulary.
        embed_dim (int): Dimensionality of token embeddings and hidden states.
        num_layers (int): Number of Transformer block layers.
        num_heads (int): Number of attention heads.
        seq_len (int): Maximum sequence length (context window).
        hidden_dim (int): Dimensionality of the feed-forward inner hidden layer.
        dropout (float, default=0.0): Dropout probability across layers.

    Example:
        >>> config = ModelConfig(
        ...     vocab_size=50257,
        ...     embed_dim=768,
        ...     num_layers=12,
        ...     num_heads=12,
        ...     seq_len=1024,
        ...     hidden_dim=3072,
        ... )
    """

    vocab_size: int
    embed_dim: int
    num_layers: int
    num_heads: int
    seq_len: int
    hidden_dim: int
    dropout: float = 0.0


@dataclass(slots=True)
class TrainingConfig:
    """Configuration container for training loop hyper-parameters.

    Args:
        max_epochs (int, default=1): Maximum number of training epochs.
        max_train_steps (int | None, default=None): Optional maximum train steps.
        max_eval_steps (int | None, default=None): Optional maximum evaluation steps.
        eval_every_n_epochs (int, default=1): Frequency of evaluation passes.
        save_every_n_epochs (int, default=1): Frequency of checkpoint saves.
        grad_accumulation_step (int, default=1): Gradient accumulation steps before optimizer step.
        max_grad_norm (float | None, default=None): Maximum gradient norm for clipping.
        lr (float, default=3e-4): Peak learning rate.
        weight_decay (float, default=0.01): Weight decay coefficient.
        optimizer_name (str, default="adamw"): Name of the optimizer ("adamw", "adam", "sgd").
        scheduler_name (str, default="none"): Name of the LR scheduler ("cosine", "linear", "none").
        warmup_steps (int, default=0): Number of warmup steps for LR scheduling.
        total_steps (int | None, default=None): Total training steps for scheduling calculation.

    Example:
        >>> train_cfg = TrainingConfig(max_epochs=5, lr=1e-4, optimizer_name="adamw")
    """

    max_epochs: int = 1
    max_train_steps: int | None = None
    max_eval_steps: int | None = None
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    grad_accumulation_step: int = 1
    max_grad_norm: float | None = None
    lr: float = 3e-4
    weight_decay: float = 0.01
    optimizer_name: str = "adamw"
    scheduler_name: str = "none"
    warmup_steps: int = 0
    total_steps: int | None = None


@dataclass(slots=True)
class GenerationConfig:
    """Configuration container for autoregressive text generation decoding.

    Args:
        max_context_len (int, default=128): Maximum context length accepted by the model.
        max_new_tokens (int, default=50): Maximum number of tokens to generate.
        temperature (float, default=1.0): Temperature scaling for logits sampling.
        top_k (int | None, default=None): Top-k filtering threshold.
        top_p (float, default=1.0): Nucleus top-p cumulative probability threshold.
        eos_token_id (int | None, default=None): End-of-sequence token ID to terminate decoding.

    Example:
        >>> gen_cfg = GenerationConfig(max_new_tokens=100, temperature=0.7, top_p=0.9)
    """

    max_context_len: int = 128
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    eos_token_id: int | None = None

