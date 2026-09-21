"""Small, reusable, model-agnostic evaluation harness for ClauseGuard.

The harness is intentionally lightweight: it reuses reliable sklearn
implementations for metrics and provides a structured, JSON-serializable result
so that the TF-IDF baseline and future models share one evaluation protocol.

Public API:

- :func:`classification_metrics` — accuracy, macro-F1, weighted-F1
- :func:`per_class_metrics` — per-class precision / recall / F1 / support
- :func:`top_k_metrics` — top-1 / top-3 / top-5 accuracy
- :func:`confusion_matrix_data` — full machine-readable confusion matrix
- :func:`top_confused_pairs` — top-N confused true/predicted label pairs
- :class:`ExperimentMetadata` — lightweight experiment metadata
- :class:`EvaluationResult` — structured, JSON-serializable result
- :func:`resolve_git_commit` — best-effort current commit hash

Deliberately NOT implemented yet (documented extension points, see
docs/evaluation.md): calibration/ECE, confidence-based abstention, and
similarity-stratified evaluation. These are deferred until the first baseline
defines how model outputs are represented.
"""

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

__all__ = [
    "classification_metrics",
    "confusion_matrix_data",
    "per_class_metrics",
    "top_confused_pairs",
    "top_k_metrics",
    "EvaluationResult",
    "ExperimentMetadata",
    "resolve_git_commit",
]