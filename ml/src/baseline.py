"""TF-IDF + Logistic Regression / LinearSVC baseline for LEDGAR clause classification.

Sections are deliberately separated so the pipeline is easy to rerun
deterministically:

- data loading            -> ``load_ledgar`` (ml.src.data_loader)
- feature extraction      -> ``build_tfidf`` / ``transform_splits``
- model training          -> ``build_logistic_regression`` / ``build_linear_svc``
- prediction              -> fitted estimator predict / predict_proba /
                             decision_function
- evaluation              -> ``ml.src.evaluation`` via ``evaluate_fitted``

The TF-IDF vectorizer is fit on the training split only; validation and test
are transformed with the same fitted vectorizer. No stop-word removal, no
stemming/lemmatization (legal words such as "shall", "may", "not", "unless",
"provided" carry meaning).

CLI commands (run from the repository root):

    python -m ml.src.baseline lr-grid     # LR validation, C x class_weight
    python -m ml.src.baseline svc-grid    # LinearSVC validation, C grid
    python -m ml.src.baseline final       # retrain on train, evaluate test once
                                          #   (--model --C --class-weight)
    python -m ml.src.baseline error-analysis  # validation error analysis
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.src.data_loader import LedgarData, load_ledgar
from ml.src.evaluation.metrics import (
    classification_metrics,
    confusion_matrix_data,
    per_class_metrics,
    top_confused_pairs,
    top_k_metrics,
)
from ml.src.evaluation.metadata import EvaluationResult, ExperimentMetadata, resolve_git_commit

SEED = 20260922
MODEL_VERSION = "baseline-v1"

TFIDF_CONFIG: dict[str, Any] = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "dtype": np.float32,
    "lowercase": True,  # matches the leakage-audit vectorizer / inference-time assumptions
}

LR_SOLVER = "lbfgs"  # L2-penalised multinomial logistic regression on sparse data
LR_MAX_ITER = 2000
# The tolerance is relaxed slightly from sklearn's default to limit the cost
# of repeated high-dimensional fits without accepting the substantial quality
# loss observed at 1e-2. Convergence and iteration counts are recorded.
LR_TOL = 1e-3
LR_C_VALUES = (1.0, 3.0, 10.0, 30.0, 100.0)
LR_CLASS_WEIGHTS = (None, "balanced")

SVC_MAX_ITER = 5000
SVC_TOL = 1e-4
SVC_C_VALUES = (0.1, 0.3, 1.0, 3.0)
SVC_CLASS_WEIGHT = None  # per spec, LinearSVC uses the shared TF-IDF representation
DEFAULT_REPORTS_DIR = "ml/reports/baseline"
DEFAULT_MODELS_DIR = "ml/artifacts"


# --------------------------------------------------------------------------- #
# Feature extraction and model builders
# --------------------------------------------------------------------------- #


def build_tfidf(config: dict[str, Any] | None = None) -> TfidfVectorizer:
    """Return a TfidfVectorizer using the fixed baseline configuration."""
    cfg = dict(TFIDF_CONFIG)
    if config:
        cfg.update(config)
    return TfidfVectorizer(**cfg)


def build_logistic_regression(
    C: float = 1.0,
    class_weight: str | None = None,
    *,
    seed: int = SEED,
) -> LogisticRegression:
    """Return a multiclass Logistic Regression for a sparse TF-IDF problem.

    ``lbfgs`` is well suited to sparse input and a hundred-class output in
    sklearn >= 1.5 (multinomial handling is automatic); ``max_iter`` is
    generous for 7-digit feature spaces.
    """
    return LogisticRegression(
        C=float(C),
        class_weight=class_weight,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        tol=LR_TOL,
        random_state=seed,
    )


def build_linear_svc(
    C: float = 1.0,
    class_weight: str | None = None,
    *,
    seed: int = SEED,
) -> LinearSVC:
    """Return a one-vs-rest linear SVM for a sparse TF-IDF problem."""
    return LinearSVC(
        penalty="l2",
        loss="squared_hinge",
        C=float(C),
        class_weight=class_weight,
        dual="auto",
        tol=SVC_TOL,
        max_iter=SVC_MAX_ITER,
        random_state=seed,
    )


def make_pipeline(
    tfidf: TfidfVectorizer | None = None,
    classifier: Any | None = None,
) -> Pipeline:
    """Combine TF-IDF vectorizer and classifier into one pipeline.

    Keeping the two together in a single artifact prevents serving from
    accidentally using a different TF-IDF configuration than training.
    The pipeline is fit once on raw text; its fitted vectorizer is reused to
    transform validation/test text during evaluation.
    """
    return Pipeline(
        [
            ("tfidf", tfidf if tfidf is not None else build_tfidf()),
            ("clf", classifier if classifier is not None else build_logistic_regression()),
        ]
    )


# --------------------------------------------------------------------------- #
# Shared feature matrix construction and evaluation
# --------------------------------------------------------------------------- #


def transform_splits(
    data: LedgarData,
    tfidf: TfidfVectorizer,
    splits: tuple[str, ...] = ("train", "validation"),
) -> dict[str, Any]:
    """Fit the vectorizer on train, transform each requested split.

    Returns ``X_<split>`` sparse matrices and ``y_<split>`` label arrays.
    """
    X_train = tfidf.fit_transform(data.splits["train"].texts)
    y_train = np.asarray(data.splits["train"].labels, dtype=int)
    matrices: dict[str, Any] = {"X_train": X_train, "y_train": y_train}
    for split in splits:
        if split == "train":
            continue
        matrices[f"X_{split}"] = tfidf.transform(data.splits[split].texts)
        matrices[f"y_{split}"] = np.asarray(data.splits[split].labels, dtype=int)
    return matrices


def _score_arrays(clf: Any, X_eval: Any, score_mode: str) -> np.ndarray:
    if score_mode == "proba":
        return np.asarray(clf.predict_proba(X_eval), dtype=float)
    if score_mode == "decision":
        return np.asarray(clf.decision_function(X_eval), dtype=float)
    raise ValueError(f"Unknown score_mode {score_mode!r}")


def _score_mode_of(clf: Any) -> str:
    return "proba" if hasattr(clf, "predict_proba") else "decision"


def evaluate_fitted(
    clf: Any,
    X_eval: Any,
    y_eval: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Evaluate a fitted classifier and return the full metric suite.

    Uses the ``ml.src.evaluation`` harness (no re-implemented metric math).
    ``y_eval`` must be integer labels; ``label_names`` provides display names
    and fixes the 100-class row/column ordering.
    """
    labels = np.arange(len(label_names))
    y_pred = clf.predict(X_eval)
    score_mode = _score_mode_of(clf)
    return {
        "metrics": classification_metrics(y_eval, y_pred, labels=labels),
        "per_class": per_class_metrics(y_eval, y_pred, labels=labels),
        "top_k": top_k_metrics(
            y_eval, _score_arrays(clf, X_eval, score_mode), labels=labels, k_values=(1, 3, 5)
        ),
        "confusion": confusion_matrix_data(y_eval, y_pred, labels=labels),
        "top_confused": top_confused_pairs(y_eval, y_pred, labels=labels, top_n=20),
    }


def build_metadata(
    data: LedgarData,
    model_name: str,
    *,
    split: str,
    configuration: dict[str, Any],
) -> ExperimentMetadata:
    return ExperimentMetadata(
        model_name=model_name,
        model_version=MODEL_VERSION,
        dataset_name=data.dataset_name,
        dataset_revision=data.version or "",
        split=split,
        # Keep the artifact portable JSON: numpy dtypes and tuples are useful
        # in sklearn constructors but are not JSON values.
        configuration=_jsonable_configuration(configuration),
        random_seed=SEED,
        git_commit=resolve_git_commit(),
    )


def _jsonable_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.__name__ if isinstance(value, type) and issubclass(value, np.generic)
            else list(value) if isinstance(value, tuple)
            else value
        )
        for key, value in configuration.items()
    }


# --------------------------------------------------------------------------- #
# Validation grid experiments
# --------------------------------------------------------------------------- #


def CG_format(C: float) -> str:
    return str(round(float(C), 4)).replace(".", "_")


def _record_grid_result(
    out_dir: Path,
    model_name: str,
    config: dict[str, Any],
    evaluated: dict[str, Any],
    data: LedgarData,
    *,
    fit_seconds: float,
    converged: bool,
    n_iter: int,
) -> dict[str, Any]:
    metadata = build_metadata(data, model_name, split="validation", configuration=config)
    result = EvaluationResult(
        metadata=metadata,
        metrics=evaluated["metrics"],
        per_class=evaluated["per_class"],
        top_k=evaluated["top_k"],
        confusion=evaluated["confusion"],
        top_confused=evaluated["top_confused"],
    )
    record = {
        **result.to_dict(),
        "fitting": {
            "fit_seconds": round(fit_seconds, 1),
            "converged": converged,
            "n_iter": n_iter,
        },
    }
    cw = config.get("class_weight")
    filename = out_dir / f"{model_name}_c{CG_format(config['C'])}_{cw or 'none'}.json"
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return record


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        cfg = rec["metadata"]["configuration"]
        rows.append(
            {
                "model": rec["metadata"]["model_name"],
                "C": cfg.get("C"),
                "class_weight": cfg.get("class_weight") or "none",
                "solver": cfg.get("solver"),
                "validation_accuracy": round(rec["metrics"]["accuracy"], 4),
                "validation_macro_f1": round(rec["metrics"]["macro_f1"], 4),
                "validation_weighted_f1": round(rec["metrics"]["weighted_f1"], 4),
                "top1_accuracy": round(rec["top_k"].get("top_1_accuracy", float("nan")), 4),
                "top3_accuracy": round(rec["top_k"].get("top_3_accuracy", float("nan")), 4),
                "top5_accuracy": round(rec["top_k"].get("top_5_accuracy", float("nan")), 4),
                "converged": rec["fitting"]["converged"],
                "n_iter": rec["fitting"]["n_iter"],
                "fit_seconds": rec["fitting"]["fit_seconds"],
            }
        )
    return rows


def _write_summary(out_dir: Path, name: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _summary_rows(records)
    with open(out_dir / f"{name}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    lines = [
        f"# {name} - validation results",
        "",
        "| model | C | class_weight | acc | macro-F1 | weighted-F1 | top1 | top3 | top5 | converged | n_iter | fit(s) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['C']} | {r['class_weight']} | {r['validation_accuracy']} | "
            f"{r['validation_macro_f1']} | {r['validation_weighted_f1']} | {r['top1_accuracy']} | "
            f"{r['top3_accuracy']} | {r['top5_accuracy']} | {r['converged']} | {r['n_iter']} | {r['fit_seconds']} |"
        )
    with open(out_dir / f"{name}_summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return rows


def run_lr_grid(data: LedgarData, out_dir: Path) -> list[dict[str, Any]]:
    """Small controlled validation experiment: C x class_weight for LR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tfidf = build_tfidf()
    matrices = transform_splits(data, tfidf)
    X_train, y_train = matrices["X_train"], matrices["y_train"]
    X_val, y_val = matrices["X_validation"], matrices["y_validation"]

    records: list[dict[str, Any]] = []
    for C, class_weight in itertools.product(LR_C_VALUES, LR_CLASS_WEIGHTS):
        config = {
            **TFIDF_CONFIG,
            "model": "logistic-regression",
            "solver": LR_SOLVER,
            "penalty": "l2",
            "C": float(C),
            "class_weight": class_weight,
            "max_iter": LR_MAX_ITER,
            "tol": LR_TOL,
        }
        clf = build_logistic_regression(C=C, class_weight=class_weight)
        start = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        n_iter = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else -1
        converged = (
            bool(np.all(np.asarray(clf.n_iter_) < LR_MAX_ITER)) if hasattr(clf, "n_iter_") else True
        )

        evaluated = evaluate_fitted(clf, X_val, y_val, data.label_names)
        record = _record_grid_result(
            out_dir, "logreg", config, evaluated, data,
            fit_seconds=fit_seconds, converged=converged, n_iter=n_iter,
        )
        records.append(record)
        print(
            f"LR C={C:g} cw={class_weight or 'none'}: acc={record['metrics']['accuracy']:.4f} "
            f"macro={record['metrics']['macro_f1']:.4f} wf1={record['metrics']['weighted_f1']:.4f} "
            f"(fit {fit_seconds:.1f}s, iters {n_iter}, converged={converged})"
        )

    _write_summary(out_dir, "logreg", records)
    return records


def run_svc_grid(data: LedgarData, out_dir: Path) -> list[dict[str, Any]]:
    """TF-IDF + LinearSVC validation experiment on the shared representation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tfidf = build_tfidf()
    matrices = transform_splits(data, tfidf)
    X_train, y_train = matrices["X_train"], matrices["y_train"]
    X_val, y_val = matrices["X_validation"], matrices["y_validation"]

    records: list[dict[str, Any]] = []
    for C in SVC_C_VALUES:
        config = {
            **TFIDF_CONFIG,
            "model": "linear-svc",
            "penalty": "l2",
            "loss": "squared_hinge",
            "C": float(C),
            "class_weight": SVC_CLASS_WEIGHT,
            "max_iter": SVC_MAX_ITER,
            "tol": SVC_TOL,
        }
        clf = build_linear_svc(C=C, class_weight=SVC_CLASS_WEIGHT)
        start = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        n_iter = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else -1
        converged = (
            bool(np.all(np.asarray(clf.n_iter_) < SVC_MAX_ITER)) if hasattr(clf, "n_iter_") else True
        )

        evaluated = evaluate_fitted(clf, X_val, y_val, data.label_names)
        record = _record_grid_result(
            out_dir, "linearsvc", config, evaluated, data,
            fit_seconds=fit_seconds, converged=converged, n_iter=n_iter,
        )
        records.append(record)
        print(
            f"SVC C={C:g}: acc={record['metrics']['accuracy']:.4f} "
            f"macro={record['metrics']['macro_f1']:.4f} wf1={record['metrics']['weighted_f1']:.4f} "
            f"(fit {fit_seconds:.1f}s, iters {n_iter}, converged={converged})"
        )

    _write_summary(out_dir, "linearsvc", records)
    return records


def load_grid_summary(reports_dir: Path, model: str) -> list[dict[str, Any]]:
    """Load the persisted validation-grid summary for a model family."""
    with open(reports_dir / f"{model}_summary.json", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Final model: retrain on train, evaluate test exactly once
# --------------------------------------------------------------------------- #


def run_final(
    data: LedgarData,
    model: str,
    C: float,
    class_weight: str | None,
    reports_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    """Retrain the selected configuration on train and evaluate test once."""
    test_report = reports_dir / "test_evaluation.json"
    if test_report.exists():
        raise FileExistsError(
            f"Refusing to evaluate test again because {test_report} already exists. "
            "Keep the first held-out result immutable."
        )
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    pipeline = make_pipeline(
        build_tfidf(),
        build_logistic_regression(C=C, class_weight=class_weight)
        if model == "logreg"
        else build_linear_svc(C=C, class_weight=class_weight),
    )
    y_train = np.asarray(data.splits["train"].labels, dtype=int)
    y_test = np.asarray(data.splits["test"].labels, dtype=int)

    start = time.perf_counter()
    pipeline.fit(data.splits["train"].texts, y_train)  # vectorizer + classifier in one step
    fit_seconds = time.perf_counter() - start

    X_test = pipeline.named_steps["tfidf"].transform(data.splits["test"].texts)
    evaluated = evaluate_fitted(pipeline.named_steps["clf"], X_test, y_test, data.label_names)

    config = {
        **TFIDF_CONFIG,
        "model": "logistic-regression" if model == "logreg" else "linear-svc",
        "C": float(C),
        "class_weight": class_weight,
        **(
            {
                "solver": LR_SOLVER,
                "penalty": "l2",
                "max_iter": LR_MAX_ITER,
                "tol": LR_TOL,
            }
            if model == "logreg"
            else {
                "penalty": "l2",
                "loss": "squared_hinge",
                "max_iter": SVC_MAX_ITER,
                "tol": SVC_TOL,
            }
        ),
    }
    config = _jsonable_configuration(config)
    metadata = build_metadata(data, f"tfidf-{model}", split="test", configuration=config)
    result = EvaluationResult(
        metadata=metadata,
        metrics=evaluated["metrics"],
        per_class=evaluated["per_class"],
        top_k=evaluated["top_k"],
        confusion=evaluated["confusion"],
        top_confused=evaluated["top_confused"],
    )
    test_record = {**result.to_dict(), "fitting": {"fit_seconds": round(fit_seconds, 1)}}
    with open(test_report, "w", encoding="utf-8") as handle:
        json.dump(test_record, handle, indent=2)

    import joblib

    artifact = models_dir / f"baseline_{model}_c{CG_format(C)}_{class_weight or 'none'}.joblib"
    joblib.dump(pipeline, artifact)
    with open(models_dir / f"{artifact.stem}.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"pipeline": str(artifact), "metadata": metadata.to_dict(), "config": config},
            handle,
            indent=2,
        )

    print(
        f"TEST [{model} C={C:g} cw={class_weight or 'none'}]: "
        f"acc={result.metrics['accuracy']:.4f} macro={result.metrics['macro_f1']:.4f} "
        f"wf1={result.metrics['weighted_f1']:.4f} top3={result.top_k.get('top_3_accuracy')} "
        f"(fit {fit_seconds:.1f}s)"
    )
    print(f"Saved pipeline -> {artifact}")
    return test_record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_lr = sub.add_parser("lr-grid", help="Logistic Regression validation grid")
    p_lr.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)

    p_svc = sub.add_parser("svc-grid", help="LinearSVC validation grid")
    p_svc.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)

    p_final = sub.add_parser("final", help="Evaluate selected config once on test")
    p_final.add_argument("--model", choices=["logreg", "linearsvc"], required=True)
    p_final.add_argument("--C", type=float, required=True)
    p_final.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    p_final.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    p_final.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)

    p_ea = sub.add_parser("error-analysis", help="Validation error analysis")
    p_ea.add_argument("--model", choices=["logreg", "linearsvc"], required=True)
    p_ea.add_argument("--C", type=float, required=True)
    p_ea.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    p_ea.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)

    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    # Keep the held-out test split out of memory during model selection and
    # validation diagnostics. Only the frozen-final command loads it.
    needed_splits = ("train", "test") if args.command == "final" else ("train", "validation")
    data = load_ledgar(splits=needed_splits)

    if args.command == "lr-grid":
        run_lr_grid(data, Path(args.reports_dir))
    elif args.command == "svc-grid":
        run_svc_grid(data, Path(args.reports_dir))
    elif args.command == "final":
        run_final(
            data,
            args.model,
            args.C,
            None if args.class_weight == "none" else "balanced",
            Path(args.reports_dir),
            Path(args.models_dir),
        )
    elif args.command == "error-analysis":
        from ml.src import error_analysis

        error_analysis.run_validation_error_analysis(
            data,
            model=args.model,
            C=args.C,
            class_weight=None if args.class_weight == "none" else "balanced",
            reports_dir=Path(args.reports_dir),
        )


if __name__ == "__main__":
    main()
