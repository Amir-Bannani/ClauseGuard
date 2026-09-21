"""Lightweight experiment metadata and structured evaluation results.

The metadata structure is intentionally small: it records the identity of a
single evaluation run so that reports are reproducible and self-describing.
Fields such as ``git_commit`` are optional and must never cause tests or CI to
fail when source-control metadata is unavailable.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_git_commit() -> str | None:
    """Return the current short commit hash, or None when unavailable.

    Best effort: returns None instead of raising when git is not installed,
    the directory is not a repository, or the subprocess fails/times out.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        commit = result.stdout.strip()
        return commit or None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


@dataclass
class ExperimentMetadata:
    """Identity metadata for one evaluation of one model.

    ``configuration`` is a free-form dict of the model/preprocessing config
    used. ``git_commit`` is optional and resolved from the repository when
    available.
    """

    model_name: str
    model_version: str = ""
    dataset_name: str = ""
    dataset_revision: str = ""
    split: str = ""
    configuration: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """A complete, JSON-serializable evaluation of one model on one split."""

    metadata: ExperimentMetadata
    metrics: dict[str, float]
    per_class: list[dict[str, Any]] = field(default_factory=list)
    top_k: dict[str, float] = field(default_factory=dict)
    confusion: dict[str, Any] = field(default_factory=dict)
    top_confused: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "metrics": self.metrics,
            "per_class": self.per_class,
            "top_k": self.top_k,
            "confusion": self.confusion,
            "top_confused": self.top_confused,
        }

    def to_json(self, path: str | Path) -> Path:
        """Serialize this result to a JSON file and return the written path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        return path