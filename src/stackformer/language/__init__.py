"""Language models package for StackFormer library.

Exposes:
    - BERT: Bidirectional Encoder Representations from Transformers
    - RoBERTa: Robustly Optimized BERT Approach model architecture
    - Decoder, GPTDecoder: Placeholder/base decoder implementations
    - EncoderDecoder: Encoder-decoder sequence-to-sequence model architecture
"""

from .decoder import Decoder, GPTDecoder
from stackformer.models.bert import BERT
from stackformer.models.roberta import RoBERTa
from .encoder_decoder import EncoderDecoder

__all__ = [
    "BERT",
    "RoBERTa",
    "Decoder",
    "GPTDecoder",
    "EncoderDecoder",
]