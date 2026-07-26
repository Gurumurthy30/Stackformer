"""Autoregressive text generation utilities for StackFormer language models.

Provides sequence generation via temperature, top-k, and top-p (nucleus) sampling with
optional KV-cache acceleration for models implementing `prefill` and `decode` interfaces.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from stackformer.config import GenerationConfig


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float,
) -> torch.Tensor:
    """Sample the next token ID from logit probabilities.

    Args:
        logits (torch.Tensor): Logits tensor of shape ``(B, V)`` for the current position.
        temperature (float): Temperature parameter > 0.
        top_k (int | None): Top-k filtering threshold.
        top_p (float): Nucleus top-p cumulative probability threshold.

    Returns:
        torch.Tensor: Sampled token IDs of shape ``(B, 1)``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if temperature != 1.0:
        logits = logits / temperature

    # Top-k sampling
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        topk_vals, topk_indices = torch.topk(logits, top_k, dim=-1)

        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(dim=-1, index=topk_indices, src=topk_vals)
        logits = filtered

    # Nucleus (top-p) sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        remove_mask = torch.zeros_like(sorted_mask)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)

        logits = logits.masked_fill(remove_mask, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _resolve_generation_config(
    max_context_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    eos_token_id: int | None,
    generation_config: GenerationConfig | None,
) -> GenerationConfig:
    """Create or resolve a GenerationConfig instance.

    Args:
        max_context_len (int): Context window length.
        max_new_tokens (int): Max tokens to generate.
        temperature (float): Temperature parameter.
        top_k (int | None): Top-k threshold.
        top_p (float): Top-p threshold.
        eos_token_id (int | None): EOS token ID.
        generation_config (GenerationConfig | None): Optional existing configuration.

    Returns:
        GenerationConfig: Resolved configuration object.
    """
    if generation_config is None:
        return GenerationConfig(
            max_context_len=max_context_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

    return generation_config

def prefill(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Prefill the KV-cache for models that support it.

    Args:
        model (torch.nn.Module): Language model instance.
        input_ids (torch.Tensor): Input token IDs of shape ``(B, T)``.

    Returns:
        tuple[torch.Tensor, dict]: Logits tensor of shape ``(B, T, V)`` and cache dictionary.
    """
    if not callable(getattr(model, "prefill", None)):
        raise RuntimeError("Model does not implement 'prefill()' method.")

    logits, cache = model.prefill(input_ids)
    return logits, cache

def decode(model: torch.nn.Module, next_token: torch.Tensor, cache: dict) -> tuple[torch.Tensor, dict]:
    """Decode the next token using the KV-cache for models that support it.

    Args:
        model (torch.nn.Module): Language model instance.
        next_token (torch.Tensor): Next token IDs of shape ``(B, 1)``.
        cache (dict): KV-cache dictionary from prefill.

    Returns:
        tuple[torch.Tensor, dict]: Logits tensor of shape ``(B, 1, V)`` and updated cache dictionary.
    """
    if not callable(getattr(model, "decode", None)):
        raise RuntimeError("Model does not implement 'decode()' method.")

    logits, updated_cache = model.decode(next_token, cache)
    return logits, updated_cache

def text_generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_context_len: int = 128,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    generation_config: GenerationConfig | None = None,
) -> torch.Tensor:
    """Autoregressively generate tokens from a language model.

    Supports standard feed-forward and fast KV-cache accelerated models.
    KV-cache models expose `supports_kv_cache = True` and implement
    `prefill(input_ids)` and `decode(next_token, cache)`.

    Args:
        model (torch.nn.Module): Language model instance.
        prompt_ids (torch.Tensor): Prompt token IDs of shape ``(B, T)`` or ``(T,)``.
        max_context_len (int, default=128): Maximum context length window.
        max_new_tokens (int, default=50): Number of tokens to generate.
        temperature (float, default=1.0): Temperature for sampling.
        top_k (int | None, default=None): Top-k filtering limit.
        top_p (float, default=1.0): Top-p nucleus filtering limit.
        eos_token_id (int | None, default=None): Optional EOS token ID to halt generation.
        generation_config (GenerationConfig | None, default=None): Optional generation config override.

    Returns:
        torch.Tensor: Generated token IDs tensor of shape ``(B, T + num_generated)``.

    Raises:
        RuntimeError: If `supports_kv_cache=True` but `prefill()` or `decode()` are missing.

    Example:
        >>> model = GPT_1(vocab_size=1000, embed_dim=128, num_layers=2, num_heads=4, seq_len=64)
        >>> prompt = torch.randint(0, 1000, (1, 10))
        >>> out = text_generate(model, prompt, max_new_tokens=5)
        >>> out.shape
        torch.Size([1, 15])
    """
    config = _resolve_generation_config(
        max_context_len=max_context_len,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        generation_config=generation_config,
    )

    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)  # (T,) -> (1, T)

    generated = prompt_ids.clone()

    batch_size = generated.size(0)
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=generated.device,
    )

    cache = None

    has_cache_methods = (
        callable(getattr(model, "prefill", None))
        and callable(getattr(model, "decode", None))
    )
    explicit_flag = getattr(model, "supports_kv_cache", None)

    if explicit_flag is True and not has_cache_methods:
        raise RuntimeError(
            "supports_kv_cache=True requires both "
            "'prefill()' and 'decode()' methods."
        )

    # Use the cache path whenever the model implements it, unless the model
    # explicitly opts out via supports_kv_cache = False.
    use_cache = has_cache_methods and explicit_flag is not False

    if use_cache:
        input_ids = generated[:, -config.max_context_len :]
        logits, cache = model.prefill(input_ids)
        logits = logits[:, -1, :]  # (B, T, V) -> (B, V)

    else:
        input_ids = generated[:, -config.max_context_len :]
        logits = model(input_ids)
        logits = logits[:, -1, :]  # (B, T, V) -> (B, V)

    for _ in range(config.max_new_tokens):

        next_token = _sample_next_token(
            logits,
            config.temperature,
            config.top_k,
            config.top_p,
        )  # (B, 1)

        if config.eos_token_id is not None:
            eos_hits = next_token.squeeze(-1) == config.eos_token_id
            finished |= eos_hits

            # Keep completed sequences fixed at EOS.
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, config.eos_token_id),
                next_token,
            )

        generated = torch.cat((generated, next_token), dim=-1)  # (B, T_curr) -> (B, T_curr + 1)

        if finished.all():
            break

        if use_cache:
            logits, cache = model.decode(next_token, cache)
            logits = logits[:, -1, :]  # (B, 1, V) -> (B, V)

        else:
            input_ids = generated[:, -config.max_context_len :]
            logits = model(input_ids)
            logits = logits[:, -1, :]  # (B, T_ctx, V) -> (B, V)

    return generated
