"""Unit tests for vectorized evaluation metrics in stackformer.training.metrics."""

import math

import pytest
import torch
import torch.nn.functional as F

from stackformer.training.metrics import (
    F1,
    Accuracy,
    Dice,
    MeanIoU,
    Perplexity,
    Precision,
    Recall,
    TopKAccuracy,
    TrainStats,
)


def test_accuracy_basic_and_edge_cases():
    acc = Accuracy(ignore_index=-100)

    # All correct
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    targets = torch.tensor([0, 1])
    acc.update(logits, (None, targets))
    assert acc.compute() == 1.0

    # Reset
    acc.reset()
    assert acc.compute() == 0.0

    # All wrong
    logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    acc.update(logits, (None, targets))
    assert acc.compute() == 0.0

    # With ignore_index
    acc.reset()
    targets_with_ignore = torch.tensor([0, -100])
    acc.update(logits, (None, targets_with_ignore))  # sample 0 is wrong, sample 1 ignored
    assert acc.compute() == 0.0


def test_accuracy_batch_additivity():
    """Updating over several batches equals one single large batch update."""
    acc_single = Accuracy()
    acc_multi = Accuracy()

    logits1 = torch.randn(4, 5)
    targets1 = torch.randint(0, 5, (4,))
    logits2 = torch.randn(6, 5)
    targets2 = torch.randint(0, 5, (6,))

    acc_multi.update(logits1, (None, targets1))
    acc_multi.update(logits2, (None, targets2))

    full_logits = torch.cat([logits1, logits2], dim=0)
    full_targets = torch.cat([targets1, targets2], dim=0)
    acc_single.update(full_logits, (None, full_targets))

    assert math.isclose(acc_single.compute(), acc_multi.compute(), abs_tol=1e-6)


def test_topk_accuracy():
    top2 = TopKAccuracy(k=2)

    # Class 1 is top-1, class 2 is top-2
    logits = torch.tensor([[0.1, 5.0, 3.0, 0.2]])
    assert top2.k == 2

    # Target is class 2 (2nd rank) -> should match top2
    top2.update(logits, (None, torch.tensor([2])))
    assert top2.compute() == 1.0

    # Target is class 3 (4th rank) -> should not match top2
    top2.reset()
    top2.update(logits, (None, torch.tensor([3])))
    assert top2.compute() == 0.0


def test_precision_recall_f1_reference():
    """Compare against hand-calculated confusion matrix values."""
    # 3 classes: 0, 1, 2
    # Predictions: [0, 0, 1, 1, 2]
    # Targets:     [0, 1, 1, 2, 2]
    # Class 0: TP=1, FP=1, FN=0 -> P=1/2=0.5, R=1/1=1.0, F1=2*(0.5*1)/(1.5)=2/3
    # Class 1: TP=1, FP=1, FN=1 -> P=1/2=0.5, R=1/2=0.5, F1=0.5
    # Class 2: TP=1, FP=0, FN=1 -> P=1/1=1.0, R=1/2=0.5, F1=2*(1*0.5)/(1.5)=2/3

    preds = torch.tensor([0, 0, 1, 1, 2])
    targets = torch.tensor([0, 1, 1, 2, 2])

    # Convert preds to one-hot / logits for update
    logits = F.one_hot(preds, num_classes=3).float() * 10.0

    # Macro Precision: (0.5 + 0.5 + 1.0) / 3 = 2.0 / 3 ≈ 0.6667
    p_macro = Precision(num_classes=3, average="macro")
    p_macro.update(logits, (None, targets))
    assert math.isclose(p_macro.compute(), 2.0 / 3.0, abs_tol=1e-4)

    # Micro Precision: TP=3, FP=2 -> 3/5 = 0.6
    p_micro = Precision(num_classes=3, average="micro")
    p_micro.update(logits, (None, targets))
    assert math.isclose(p_micro.compute(), 0.6, abs_tol=1e-4)

    # Macro Recall: (1.0 + 0.5 + 0.5) / 3 = 2.0 / 3 ≈ 0.6667
    r_macro = Recall(num_classes=3, average="macro")
    r_macro.update(logits, (None, targets))
    assert math.isclose(r_macro.compute(), 2.0 / 3.0, abs_tol=1e-4)

    # Macro F1: (2/3 + 0.5 + 2/3) / 3 = (1.3333 + 0.5) / 3 = 1.8333 / 3 = 0.6111
    f1_macro = F1(num_classes=3, average="macro")
    f1_macro.update(logits, (None, targets))
    expected_f1 = (2 / 3 + 0.5 + 2 / 3) / 3
    assert math.isclose(f1_macro.compute(), expected_f1, abs_tol=1e-4)


def test_precision_with_absent_class():
    """Class absent from predictions should not cause division by zero."""
    p_macro = Precision(num_classes=4, average="macro")
    # Only classes 0 and 1 present
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
    targets = torch.tensor([0, 1])
    p_macro.update(logits, (None, targets))
    # Class 2 and 3 have denom=0, so excluded from macro average: (1.0 + 1.0) / 2 = 1.0
    assert p_macro.compute() == 1.0


def test_perplexity_matches_shifted_cross_entropy():
    """Perplexity in LM mode must equal exp(mean shifted CE)."""
    torch.manual_seed(42)
    B, T, V = 2, 6, 10
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))

    # Shifted CE manually
    shift_logits = logits[:, :-1, :].reshape(-1, V)
    shift_targets = targets[:, 1:].reshape(-1)
    manual_ce = F.cross_entropy(shift_logits, shift_targets)
    expected_ppl = math.exp(manual_ce.item())

    ppl = Perplexity(loss_mode="lm")
    ppl.update(logits, (None, targets))
    computed_ppl = ppl.compute()

    assert math.isclose(computed_ppl, expected_ppl, rel_tol=1e-4)


def test_segmentation_mean_iou_and_dice():
    # 2 classes: 0 and 1
    # 1 image of 4 pixels:
    # Pred:   [0, 1, 1, 0]
    # Target: [0, 1, 0, 1]
    # Class 0:
    #   Pred: 0, 3. Target: 0. Intersect: {0} (1). Union: {0, 3} (2). IoU = 1/2 = 0.5
    # Class 1:
    #   Pred: 1, 2. Target: 1, 3. Intersect: {1} (1). Union: {1, 2, 3} (3). IoU = 1/3 ≈ 0.3333
    # Mean IoU = (0.5 + 1/3) / 2 = 5/12 ≈ 0.4167
    preds = torch.tensor([[[0, 1], [1, 0]]])
    targets = torch.tensor([[[0, 1], [0, 1]]])

    miou = MeanIoU(num_classes=2)
    miou.update(preds, (None, targets))
    assert math.isclose(miou.compute(), 1.0 / 3.0, abs_tol=1e-4)

    dice = Dice(num_classes=2)
    dice.update(preds, (None, targets))
    assert math.isclose(dice.compute(), 0.5, abs_tol=1e-4)


def test_train_stats_helper():
    stats = TrainStats()
    stats.start_step()
    res = stats.end_step(samples=32, tokens=1024, loss=2.5, lr=1e-4, grad_norm=0.8)

    assert "step_time_sec" in res
    assert "samples_per_sec" in res
    assert "tokens_per_sec" in res
    assert res["train_loss"] == 2.5
    assert res["lr"] == 1e-4
    assert res["grad_norm"] == 0.8
    assert res["samples_per_sec"] > 0
    assert res["tokens_per_sec"] > 0
