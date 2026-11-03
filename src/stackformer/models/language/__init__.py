"""Language model architectures and building blocks.

Exposes:
    - BERT: Bidirectional Encoder Representations from Transformers
    - RoBERTa: Robustly Optimized BERT Approach model architecture
    - GPT1, GPT2: Generative Pre-trained Transformer models
    - Llama1, Llama2: Meta LLaMA autoregressive models
    - Gemma1_2B, Gemma1_7B: Google Gemma open models
    - Transformer: Original encoder-decoder sequence-to-sequence model
    - Decoder, GPTDecoder: Abstract autoregressive decoder base classes
    - EncoderDecoder: Encoder-decoder sequence-to-sequence base class
"""

from .bert import BERT
from .decoder import Decoder, GPTDecoder
from .encoder_decoder import EncoderDecoder
from .gemma import (
    Gemma1_2B,
    Gemma1_7B,
    Gemma_1_2B,
    Gemma_1_7B,
    gemma_1_2b,
    gemma_1_7b,
)
from .gpt import GPT1, GPT2, GPT_1, GPT_2
from .llama import (
    Llama1,
    Llama2,
    Llama_1,
    Llama_2,
    llama_1,
    llama_2,
)
from .roberta import RoBERTa
from .transformer import Transformer

__all__ = [
    "BERT",
    "Decoder",
    "EncoderDecoder",
    "GPT1",
    "GPT2",
    "GPTDecoder",
    "GPT_1",
    "GPT_2",
    "Gemma1_2B",
    "Gemma1_7B",
    "Gemma_1_2B",
    "Gemma_1_7B",
    "Llama1",
    "Llama2",
    "Llama_1",
    "Llama_2",
    "RoBERTa",
    "Transformer",
    "gemma_1_2b",
    "gemma_1_7b",
    "llama_1",
    "llama_2",
]
