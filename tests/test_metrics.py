"""Tests for the small model-agnostic evaluation harness (synthetic data only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.src.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_data,
    per_class_metrics,
    top_confused_pairs,
    top_k_metrics,
)
from ml.src.evaluation.metadata import (
    EvaluationResult,
    ExperimentMetadata,
    resolve_git_commit,
)


def _demo_predictions():
    # 6 samples, 4 classes. Predictions are imperfect on purpose.
    y_true = [0, 1, 2, 3, 0, 1]
    y_pred = [0, 1, 2, 3, 1, 0]
    return y_true, y_pred


def _demo_scores():
    # Probabilities / decision scores aligned to labels [0, 1, 2, 3].
    # Row i gives the true class the highest score.
    return np.array(
        [
            [0.7, 0.2, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.1, 0.75, 0.05],
            [0.1, 0.1, 0.1, 0.7],
            [0.6, 0.3, 0.05, 0.05],
            [0.55, 0.35, 0.05, 0.05],
        ],
        dtype=float,
    )


def test_accuracy() -> None:
    y_true = [0, 1, 0, 2, 1]
    y_pred = [0, 1, 0, 1, 1]
    result = classification_metrics(y_true, y_pred)
    assert result["accuracy"] == pytest.approx(4 / 5)


def test_macro_and_weighted_f1() -> None:
    y_true = [0, 0, 0, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    result = classification_metrics(y_true, y_pred)

    # Macro-F1 averages per-class F1 equally:
    # class 0: tp=2, fp=0, fn=1 -> P=1, R=2/3 -> F1=4/5
    # class 1: tp=1, fp=1, fn=0 -> P=1/2, R=1 -> F1=2/3
    # class 2: tp=2 -> F1=1
    assert result["macro_f1"] == pytest.approx((4 / 5 + 2 / 3 + 1) / 3)

    # Weighted-F1 weights per-class F1 by support (3, 1, 2).
    assert result["weighted_f1"] == pytest.approx((4 / 5 * 3 + 2 / 3 * 1 + 1 * 2) / 6)


def test_zero_division_handling() -> None:
    # Class 2 never appears in predictions; with zero_division=0 its F1 is 0.
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0

    per_class = per_class_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert per_class[2] == {"label": 2, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def test_per_class_metrics() -> None:
    y_true, y_pred = _demo_predictions()
    rows = per_class_metrics(y_true, y_pred, labels=[0, 1, 2, 3])

    assert [row["label"] for row in rows] == [0, 1, 2, 3]
    assert [row["support"] for row in rows] == [2, 2, 1, 1]

    # Class 0: predicts 0 at indices 0 and 5; true class-0 at indices 0 and 4.
    # Recall: tp=1, fn=1 -> 1/2. Precision: tp=1, fp=1 (index 5) -> 1/2.
    assert rows[0]["recall"] == pytest.approx(1 / 2)
    assert rows[0]["precision"] == pytest.approx(1 / 2)


def test_per_class_metrics_includes_zero_support_labels() -> None:
    y_true = [0, 1]
    y_pred = [0, 1]
    rows = per_class_metrics(y_true, y_pred, labels=[0, 1, 5])
    assert len(rows) == 3
    assert rows[2]["label"] == 5
    assert rows[2]["support"] == 0


def test_top_k_accuracy_with_scores() -> None:
    y_true = [0, 1, 2, 3, 0, 1]
    scores = _demo_scores()
    result = top_k_metrics(y_true, scores, labels=[0, 1, 2, 3], k_values=(1, 3, 5))

    assert result["top_1_accuracy"] == pytest.approx(5 / 6)  # index 1 misranked
    assert result["top_3_accuracy"] == pytest.approx(1.0)
    # top-5 not computable with 4 score columns -> omitted.
    assert "top_5_accuracy" not in result


def test_top_k_accepts_decision_scores() -> None:
    # Same ranking, but values are unbounded decision scores (e.g. LinearSVC).
    y_true = [0, 1, 2, 3, 0, 1]
    scores = _demo_scores() * 10.0 - 3.0
    result = top_k_metrics(y_true, scores, labels=[0, 1, 2, 3], k_values=(1,))
    assert result["top_1_accuracy"] == pytest.approx(5 / 6)


def test_top_k_accuracy_perfect() -> None:
    y_true = [0, 1, 2, 3, 0, 1]
    scores = np.array(
        [
            [0.90, 0.05, 0.02, 0.03],
            [0.05, 0.90, 0.02, 0.03],
            [0.05, 0.02, 0.90, 0.03],
            [0.05, 0.02, 0.03, 0.90],
            [0.90, 0.05, 0.02, 0.03],
            [0.05, 0.90, 0.02, 0.03],
        ],
        dtype=float,
    )
    result = top_k_metrics(y_true, scores, labels=[0, 1, 2, 3], k_values=(1, 3))
    assert result["top_1_accuracy"] == 1.0
    assert result["top_3_accuracy"] == 1.0


def test_top_k_requires_aligned_labels() -> None:
    y_true = [0, 1, 2, 3, 0, 1]
    scores = _demo_scores()
    with pytest.raises(ValueError):
        top_k_metrics(y_true, scores, labels=[0, 1, 2], k_values=(1,))


def test_top_k_rejects_bad_shapes() -> None:
    y_true = [0, 1, 2]
    # Flattened scores are not 2-D.
    with pytest.raises(ValueError):
        top_k_metrics(y_true, np.array([1.0, 2.0]), labels=[0, 1], k_values=(1,))


def test_confusion_matrix_data() -> None:
    y_true, y_pred = _demo_predictions()
    data = confusion_matrix_data(y_true, y_pred, labels=[0, 1, 2, 3])
    assert data["labels"] == [0, 1, 2, 3]
    matrix = np.asarray(data["matrix"])
    assert matrix.shape == (4, 4)
    # Diagonal: true==pred for indices 0,1,2,3; 4 and 5 are confused.
    assert matrix.tolist() == [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert int(matrix.sum()) == len(y_true)


def test_top_confused_pairs_excludes_diagonal_by_default() -> None:
    y_true, y_pred = _demo_predictions()
    pairs = top_confused_pairs(y_true, y_pred, labels=[0, 1, 2, 3], top_n=10)
    for pair in pairs:
        assert pair["true_label"] != pair["predicted_label"]
    counts = {(p["true_label"], p["predicted_label"]): p["count"] for p in pairs}
    assert counts == {(0, 1): 1, (1, 0): 1}
    # Sorted by count desc, then true_label asc -> the (0,1) pair comes first.
    assert pairs[0]["true_label"] == 0
    assert pairs[0]["predicted_label"] == 1


def test_top_confused_pairs_include_diagonal() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    pairs = top_confused_pairs(y_true, y_pred, labels=[0, 1], top_n=10, include_diagonal=True)
    assert pairs[0] == {"true_label": 0, "predicted_label": 0, "count": 2}


def test_metadata_defaults_are_optional() -> None:
    metadata = ExperimentMetadata(model_name="tfidf-logreg")
    assert metadata.model_version == ""
    assert metadata.git_commit is None
    assert metadata.to_dict()["model_name"] == "tfidf-logreg"


def test_metadata_git_commit_resolution() -> None:
    commit = resolve_git_commit()
    # In a repo it must be a string; outside a repo it degrades to None.
    assert commit is None or (isinstance(commit, str) and len(commit) > 0)


def test_evaluation_result_json_round_trip(tmp_path) -> None:
    metadata = ExperimentMetadata(
        model_name="dummy",
        dataset_name="coastalcph/lex_glue",
        configuration={"ngram_range": [1, 2]},
    )
    result = EvaluationResult(
        metadata=metadata,
        metrics={"accuracy": 0.5, "macro_f1": 0.4, "weighted_f1": 0.45},
        per_class=[{"label": 0, "precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}],
        top_k={"top_1_accuracy": 0.5},
        confusion={"labels": [0, 1], "matrix": [[1, 0], [1, 0]]},
        top_confused=[{"true_label": 1, "predicted_label": 0, "count": 1}],
    )

    out = tmp_path / "evaluation.json"
    written = result.to_json(out)
    assert written == out
    assert out.exists()

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["metadata"]["model_name"] == "dummy"
    assert loaded["metrics"]["macro_f1"] == 0.4
    assert loaded["top_k"]["top_1_accuracy"] == 0.5
    assert loaded["confusion"]["matrix"] == [[1, 0], [1, 0]]