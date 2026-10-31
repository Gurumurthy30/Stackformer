"""Models package for StackFormer library architectures.

Exposes:
    - GPT1, GPT2 (GPT_1, GPT_2): OpenAI GPT model constructors
    - Gemma1_2B, Gemma1_7B (gemma_1_2b, gemma_1_7b, Gemma_1_2B, Gemma_1_7B): Google Gemma architecture constructors
    - Llama1, Llama2 (llama_1, llama_2, Llama_1, Llama_2): Meta LLaMA architecture constructors
    - BERT, RoBERTa: Bidirectional encoder language models
    - Transformer: Standard Encoder-Decoder Transformer model architecture
"""

from .bert import BERT
from .gemma import Gemma1_2B, Gemma1_7B, Gemma_1_2B, Gemma_1_7B, gemma_1_2b, gemma_1_7b
from .gpt import GPT1, GPT2, GPT_1, GPT_2
from .llama import Llama1, Llama2, Llama_1, Llama_2, llama_1, llama_2
from .roberta import RoBERTa
from .transformer import Transformer

__all__ = [
    "BERT",
    "GPT1",
    "GPT2",
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
