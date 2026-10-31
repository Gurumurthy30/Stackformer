"""Public metric function exports for StackFormer models and evaluation pipelines.

Exposes:
    - accuracy: Calculate classification accuracy
    - perplexity: Calculate language model perplexity from cross-entropy loss
    - precision, recall, f1_score, precision_recall_f1: Binary classification metric functions
    - distributed_mean: Compute distributed tensor mean across DDP ranks
"""

from stackformer.logging.metrics import (
    accuracy,
    distributed_mean,
    f1_score,
    perplexity,
    precision,
    precision_recall_f1,
    recall,
)

__all__ = [
    "accuracy",
    "perplexity",
    "precision",
    "recall",
    "f1_score",
    "precision_recall_f1",
    "distributed_mean",
]

