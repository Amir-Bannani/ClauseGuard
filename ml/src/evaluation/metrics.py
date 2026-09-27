"""Core evaluation metrics for ClauseGuard clause classifiers.

This module is deliberately small and model-agnostic. It reuses reliable sklearn
implementations instead of reimplementing metric mathematics:

- accuracy
- macro-F1 (primary metric for the imbalanced 100-class LEDGAR task)
- weighted-F1 (support-weighted view)
- per-class precision / recall / F1 / support
- top-k accuracy, accepting either probability arrays or decision-score arrays
- full confusion matrix and top-N confused true/predicted pairs

Decision-score arrays (e.g. ``LinearSVC.decision_function``) and probability
arrays (e.g. ``LogisticRegression.predict_proba``) are interchangeable inputs
for :func:`top_k_metrics` because ``sklearn.metrics.top_k_accuracy_score`` only
requires real-valued scores per class.

Not implemented here (future production-oriented capabilities, see
docs/evaluation.md): calibration/ECE, confidence-based abstention, and
similarity-stratified evaluation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

# Single-label multiclass: micro-F1 == accuracy, so micro-F1 is intentionally
# not exposed as a separate primary metric (redundant information).
DEFAULT_K_VALUES = (1, 3, 5)


def _resolve_labels(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """Return the sorted union of labels appearing in targets and predictions."""
    return np.sort(
        np.union1d(np.unique(np.asarray(y_true)), np.unique(np.asarray(y_pred)))
    ).astype(int)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
    *,
    zero_division: int = 0,
) -> dict[str, float]:
    """Return accuracy, macro-F1 and weighted-F1 as a dict.

    ``zero_division=0`` is used so classes with no predicted examples yield 0
    rather than raising or returning NaN.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                labels=labels,
                average="macro",
                zero_division=zero_division,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                labels=labels,
                average="weighted",
                zero_division=zero_division,
            )
        ),
    }


def per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
    *,
    zero_division: int = 0,
) -> list[dict[str, Any]]:
    """Return per-class precision / recall / F1 / support.

    ``labels`` fixes the class ordering (and includes classes with zero
    support, e.g. all 100 LEDGAR classes). When None, the sorted union of
    observed labels is used.
    """
    if labels is None:
        label_arr = _resolve_labels(y_true, y_pred)
    else:
        label_arr = np.asarray(labels, dtype=int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_arr, zero_division=zero_division
    )

    rows = []
    for idx, label in enumerate(label_arr):
        rows.append(
            {
                "label": int(label),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    return rows


def top_k_metrics(
    y_true: Sequence[int],
    scores: Any,
    labels: Sequence[int] | None = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Return top-k accuracy for each requested ``k``.

    ``scores`` must be a 2-D array of shape (n_samples, n_classes) containing
    class scores (probabilities or decision scores). Column order must match
    the ordering of ``labels``. Every requested k larger than the number of
    available score columns is skipped (no score column to rank against).
    """
    scores_arr = np.asarray(scores, dtype=float)
    if scores_arr.ndim != 2:
        raise ValueError("scores must be a 2-D array of shape (n_samples, n_classes)")
    if scores_arr.shape[0] != len(y_true):
        raise ValueError(
            "scores must have one row per sample in y_true "
            f"({scores_arr.shape[0]} != {len(y_true)})"
        )

    if labels is None:
        label_arr = np.sort(np.unique(np.asarray(y_true))).astype(int)
    else:
        label_arr = np.asarray(labels, dtype=int)
    if label_arr.size != scores_arr.shape[1]:
        raise ValueError(
            "labels must have one entry per score column "
            f"({label_arr.size} != {scores_arr.shape[1]})"
        )

    result: dict[str, float] = {}
    for k in k_values:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > scores_arr.shape[1]:
            # Cannot rank more classes than there are score columns.
            continue
        result[f"top_{k}_accuracy"] = float(
            top_k_accuracy_score(
                y_true, scores_arr, k=k, labels=label_arr, normalize=True
            )
        )
    return result


def confusion_matrix_data(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return a machine-readable full confusion matrix.

    Format: ``{"labels": [0, 1, ...], "matrix": [[row, ...], ...]}`` where rows
    are true labels and columns are predicted labels (both ordered by
    ``labels``).
    """
    if labels is None:
        label_arr = _resolve_labels(y_true, y_pred)
    else:
        label_arr = np.asarray(labels, dtype=int)

    matrix = confusion_matrix(y_true, y_pred, labels=label_arr)
    return {
        "labels": [int(label) for label in label_arr],
        "matrix": matrix.astype(int).tolist(),
    }


def top_confused_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
    top_n: int = 10,
    *,
    include_diagonal: bool = False,
) -> list[dict[str, Any]]:
    """Return the top-N confused true/predicted label pairs sorted by count.

    By default the diagonal (correct predictions) is excluded so the output
    highlights genuine confusions. Pass ``include_diagonal=True`` to include
    correct predictions as well.
    """
    if labels is None:
        label_arr = _resolve_labels(y_true, y_pred)
    else:
        label_arr = np.asarray(labels, dtype=int)

    matrix = confusion_matrix(y_true, y_pred, labels=label_arr)
    pairs: list[dict[str, Any]] = []
    for i, true_label in enumerate(label_arr):
        for j, pred_label in enumerate(label_arr):
            if not include_diagonal and i == j:
                continue
            count = int(matrix[i, j])
            if count == 0:
                continue
            pairs.append(
                {
                    "true_label": int(true_label),
                    "predicted_label": int(pred_label),
                    "count": count,
                }
            )

    pairs.sort(key=lambda pair: (-pair["count"], pair["true_label"], pair["predicted_label"]))
    return pairs[:top_n]
