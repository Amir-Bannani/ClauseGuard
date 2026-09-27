"""Validation error analysis for the selected baseline configuration.

Diagnostic only - the goal is to understand where the baseline fails on
validation, not to drive another optimization loop. No test-set data is used.

Analyses produced (all under ``<reports_dir>/error_analysis/``):

1. top-confused pair summary (with example snippets)
2. per-class performance vs class support (rare-class focus)
3. confident errors (high-confidence, wrong predictions)
4. performance by clause character-length bucket
5. performance by similarity-to-training bucket (reuses the audit's
   ``val_similarity_to_train.npy`` when available)
6. a seeded sample of misclassified validation examples for manual inspection

Run from the repository root:
    python -m ml.src.baseline error-analysis --model logreg --C 10 --class-weight none
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.src.baseline import (
    SEED,
    build_linear_svc,
    build_logistic_regression,
    build_tfidf,
    make_pipeline,
)
from ml.src.data_loader import LedgarData
from ml.src.evaluation.metrics import (
    classification_metrics,
    per_class_metrics,
    top_confused_pairs,
)

SNIPPET_LEN = 220
CONFIDENT_ERRORS_TOP_N = 20
ERROR_SAMPLE_N = 30
LENGTH_BUCKETS = [
    ("<200", 0, 200),
    ("200-499", 200, 500),
    ("500-999", 500, 1000),
    ("1000-1999", 1000, 2000),
    (">=2000", 2000, float("inf")),
]
SIMILARITY_BUCKETS = [
    ("<0.50", 0.0, 0.50),
    ("0.50-0.80", 0.50, 0.80),
    ("0.80-0.90", 0.80, 0.90),
    ("0.90-0.95", 0.90, 0.95),
    ("0.95-0.98", 0.95, 0.98),
    (">=0.98", 0.98, float("inf")),
]
DEFAULT_SIMILARITY_FILE = Path("ml/reports/val_similarity_to_train.npy")


def _snippet(text: str, limit: int = SNIPPET_LEN) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


def _fit_for_validation(
    data: LedgarData, model: str, C: float, class_weight: str | None
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the selected config on train, transform validation, return pieces."""
    pipeline = make_pipeline(
        build_tfidf(),
        build_logistic_regression(C=C, class_weight=class_weight)
        if model == "logreg"
        else build_linear_svc(C=C, class_weight=class_weight),
    )
    y_train = np.asarray(data.splits["train"].labels, dtype=int)
    y_val = np.asarray(data.splits["validation"].labels, dtype=int)
    pipeline.fit(data.splits["train"].texts, y_train)
    X_val = pipeline.named_steps["tfidf"].transform(data.splits["validation"].texts)
    clf = pipeline.named_steps["clf"]
    y_pred = np.asarray(clf.predict(X_val), dtype=int)
    if hasattr(clf, "predict_proba"):
        scores = np.asarray(clf.predict_proba(X_val), dtype=float)
    else:
        scores = np.asarray(clf.decision_function(X_val), dtype=float)
    return clf, X_val, y_val, y_pred, scores


def _confident_errors(
    y_val: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, texts: list[str], label_names: list[str]
) -> list[dict[str, Any]]:
    wrong = y_pred != y_val
    if not np.any(wrong):
        return []
    confidence = np.max(scores, axis=1)
    idx = np.where(wrong)[0]
    order = idx[np.argsort(-confidence[idx])]
    rows = []
    for i in order[:CONFIDENT_ERRORS_TOP_N]:
        rows.append(
            {
                "ds_index": int(i),
                "true_label": int(y_val[i]),
                "true_label_name": label_names[int(y_val[i])],
                "pred_label": int(y_pred[i]),
                "pred_label_name": label_names[int(y_pred[i])],
                "confidence": float(confidence[i]),
                "snippet": _snippet(texts[i]),
            }
        )
    return rows


def _length_bucket_analysis(
    texts: list[str], y_val: np.ndarray, y_pred: np.ndarray
) -> list[dict[str, Any]]:
    lengths = np.asarray([len(t) for t in texts], dtype=int)
    rows = []
    for name, low, high in LENGTH_BUCKETS:
        if high == float("inf"):
            mask = lengths >= low
        else:
            mask = (lengths >= low) & (lengths < high)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": name, "n": 0})
            continue
        acc = float(np.mean(y_pred[mask] == y_val[mask]))
        rows.append(
            {
                "bucket": name,
                "n": n,
                "correct": int(np.sum(y_pred[mask] == y_val[mask])),
                "accuracy": round(acc, 4),
            }
        )
    return rows


def _similarity_stratification(
    y_val: np.ndarray,
    y_pred: np.ndarray,
    similarity_file: Path,
    num_classes: int,
) -> list[dict[str, Any]] | None:
    if not similarity_file.exists():
        return None
    sims = np.load(str(similarity_file)).astype(float)
    if sims.shape[0] != len(y_val):
        raise ValueError(
            f"Similarity array shape {sims.shape} does not match validation size {len(y_val)}"
        )
    rows = []
    for name, low, high in SIMILARITY_BUCKETS:
        if high == float("inf"):
            mask = sims >= low
        else:
            mask = (sims >= low) & (sims < high)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": name, "n": 0})
            continue
        y_true_b = y_val[mask]
        y_pred_b = y_pred[mask]
        metrics = classification_metrics(
            y_true_b, y_pred_b, labels=np.arange(num_classes)
        )
        rows.append(
            {
                "bucket": name,
                "n": n,
                "correct": int(np.sum(y_true_b == y_pred_b)),
                "accuracy": round(metrics["accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "unique_labels": int(np.unique(y_true_b).size),
            }
        )
    return rows


def _per_class_vs_support(
    y_val: np.ndarray, y_pred: np.ndarray, data: LedgarData
) -> list[dict[str, Any]]:
    from collections import Counter

    training_support = Counter(data.splits["train"].labels)

    per_class = per_class_metrics(
        y_val, y_pred, labels=np.arange(data.num_classes)
    )
    rows = []
    for row in per_class:
        rows.append(
            {
                "label": row["label"],
                "label_name": data.label_names[row["label"]],
                "val_support": row["support"],
                "train_support": training_support[row["label"]],
                "precision": round(row["precision"], 4),
                "recall": round(row["recall"], 4),
                "f1": round(row["f1"], 4),
                "rare": training_support[row["label"]] < 100,
            }
        )
    rows.sort(key=lambda r: (r["f1"], r["label"]))
    return rows


def _top_confused_with_examples(
    y_val: np.ndarray,
    y_pred: np.ndarray,
    texts: list[str],
    label_names: list[str],
    top_n: int = 10,
    examples_per_pair: int = 3,
) -> list[dict[str, Any]]:
    pairs = top_confused_pairs(
        y_val, y_pred, labels=np.arange(len(label_names)), top_n=top_n
    )
    out = []
    for pair in pairs:
        true_id, pred_id = pair["true_label"], pair["predicted_label"]
        indices = np.where((y_val == true_id) & (y_pred == pred_id))[0]
        examples = [
            {"ds_index": int(i), "snippet": _snippet(texts[i])}
            for i in indices[:examples_per_pair]
        ]
        out.append(
            {
                "true_label": true_id,
                "true_label_name": label_names[true_id],
                "predicted_label": pred_id,
                "predicted_label_name": label_names[pred_id],
                "count": pair["count"],
                "examples": examples,
            }
        )
    return out


def _sample_errors(
    y_val: np.ndarray,
    y_pred: np.ndarray,
    texts: list[str],
    label_names: list[str],
    n: int = ERROR_SAMPLE_N,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    wrong = np.where(y_val != y_pred)[0]
    if not len(wrong):
        return []
    rng = np.random.default_rng(seed)
    sample = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
    sample = np.sort(sample)
    return [
        {
            "ds_index": int(i),
            "true_label_name": label_names[int(y_val[i])],
            "pred_label_name": label_names[int(y_pred[i])],
            "snippet": _snippet(texts[i], limit=400),
        }
        for i in sample
    ]


def _write_markdown(directory: Path, filename: str, lines: list[str]) -> None:
    with open(directory / filename, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_validation_error_analysis(
    data: LedgarData,
    model: str,
    C: float,
    class_weight: str | None,
    reports_dir: Path,
    similarity_file: Path = DEFAULT_SIMILARITY_FILE,
) -> dict[str, Any]:
    """Run all diagnostic analyses on the validation set for one configuration."""
    out = reports_dir / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)

    clf, _, y_val, y_pred, scores = _fit_for_validation(data, model, C, class_weight)
    texts = data.splits["validation"].texts
    label_names = data.label_names

    overview = classification_metrics(
        y_val, y_pred, labels=np.arange(data.num_classes)
    )
    print(
        f"Error analysis for {model} C={C} cw={class_weight or 'none'}: "
        f"acc={overview['accuracy']:.4f} macro={overview['macro_f1']:.4f}"
    )

    artifacts: dict[str, Any] = {
        "model": model,
        "C": C,
        "class_weight": class_weight,
        "overview": overview,
    }

    # 1. Top confused pairs with examples
    confused = _top_confused_with_examples(y_val, y_pred, texts, label_names)
    artifacts["top_confused"] = confused
    with open(out / "top_confused.json", "w", encoding="utf-8") as handle:
        json.dump(confused, handle, indent=2)

    lines = ["# Top confused class pairs (validation)", ""]
    for pair in confused:
        lines.append(
            f"## {pair['true_label_name']} (true) -> {pair['predicted_label_name']} (pred): {pair['count']}"
        )
        for ex in pair["examples"]:
            lines.append(f"- val[{ex['ds_index']}]: ``{ex['snippet']}``")
        lines.append("")
    _write_markdown(out, "top_confused.md", lines)

    # 2. Per-class vs support
    per_class = _per_class_vs_support(y_val, y_pred, data)
    artifacts["per_class"] = per_class
    with open(out / "per_class_vs_support.json", "w", encoding="utf-8") as handle:
        json.dump(per_class, handle, indent=2)

    lines = ["# Per-class performance vs support (validation)", "",
             "| label | name | val_support | train_support | P | R | F1 | rare |",
             "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in per_class:
        lines.append(
            f"| {row['label']} | {row['label_name']} | {row['val_support']} | "
            f"{row['train_support']} | {row['precision']} | {row['recall']} | {row['f1']} | "
            f"{'yes' if row['rare'] else ''} |"
        )
    _write_markdown(out, "per_class_vs_support.md", lines)

    # 3. Confident errors
    confident = _confident_errors(y_val, y_pred, scores, texts, label_names)
    artifacts["confident_errors"] = confident
    with open(out / "confident_errors.json", "w", encoding="utf-8") as handle:
        json.dump(confident, handle, indent=2)

    # 4. Length buckets
    length_rows = _length_bucket_analysis(texts, y_val, y_pred)
    artifacts["length_buckets"] = length_rows
    with open(out / "length_buckets.json", "w", encoding="utf-8") as handle:
        json.dump(length_rows, handle, indent=2)

    # 5. Similarity stratification (reuses the audit artifacts when present)
    sim_rows = _similarity_stratification(
        y_val, y_pred, similarity_file, data.num_classes
    )
    artifacts["similarity_strata"] = sim_rows
    with open(out / "similarity_strata.json", "w", encoding="utf-8") as handle:
        json.dump(sim_rows, handle, indent=2)

    # 6. Seeded error sample for manual inspection
    error_sample = _sample_errors(y_val, y_pred, texts, label_names)
    artifacts["error_sample"] = error_sample
    with open(out / "error_sample.json", "w", encoding="utf-8") as handle:
        json.dump(error_sample, handle, indent=2)

    lines = [f"# Error-analysis summary ({model} C={C} cw={class_weight or 'none'}, validation)",
             "",
             f"- accuracy: {overview['accuracy']:.4f}",
             f"- macro-F1: {overview['macro_f1']:.4f}",
             f"- weighted-F1: {overview['weighted_f1']:.4f}",
             f"- errors: {int(np.sum(y_val != y_pred)):,} / {len(y_val):,}",
             "",
             "## Accuracy by clause length",
             "",
             "| bucket | n | correct | accuracy |",
             "| --- | ---: | ---: | ---: |",
             ]
    for row in length_rows:
        if row["n"]:
            lines.append(f"| {row['bucket']} | {row['n']:,} | {row['correct']} | {row['accuracy']} |")
    if sim_rows is not None:
        lines += ["", "## Accuracy by similarity to training set", "",
                  "| bucket | n | correct | accuracy | macro-F1 | unique_labels |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in sim_rows:
            if row["n"]:
                lines.append(
                    f"| {row['bucket']} | {row['n']:,} | {row['correct']} | "
                    f"{row['accuracy']} | {row['macro_f1']} | {row['unique_labels']} |"
                )
    lines += ["", "## Confident errors (wrong, high confidence)", ""]
    for row in confident[:10]:
        lines.append(
            f"- val[{row['ds_index']}] {row['true_label_name']} -> {row['pred_label_name']} "
            f"(conf {row['confidence']:.3f}): ``{row['snippet']}``"
        )
    _write_markdown(out, "summary.md", lines)

    with open(out / "all_artifacts.json", "w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, indent=2)

    print(f"Error analysis artifacts saved to {out}")
    return artifacts


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Minimal CLI convenience; the normal entry point is
    # `python -m ml.src.baseline error-analysis ...`.
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["logreg", "linearsvc"], required=True)
    parser.add_argument("--C", type=float, required=True)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--reports-dir", default="ml/reports/baseline")
    parser.add_argument("--similarity-file", default=str(DEFAULT_SIMILARITY_FILE))
    args = parser.parse_args()
    from ml.src.data_loader import load_ledgar

    data = load_ledgar(splits=("train", "validation"))
    run_validation_error_analysis(
        data,
        model=args.model,
        C=args.C,
        class_weight=None if args.class_weight == "none" else "balanced",
        reports_dir=Path(args.reports_dir),
        similarity_file=Path(args.similarity_file),
    )
