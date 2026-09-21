"""Audit exact overlap and TF-IDF near-duplicate similarity across dataset splits.

Run from repository root:
    python ml/src/audit_leakage.py [--output-dir ml/reports] [--batch-size 1000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Allow running this file directly from the repository root as a script:
#     python ml/src/audit_leakage.py
# When run via `python -m pytest` the repository root is already on sys.path,
# so this insertion is a no-op.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.src.normalization import normalize_text


DATASET_NAME = "coastalcph/lex_glue"
DATASET_CONFIG = "ledgar"
DEFAULT_OUTPUT_DIR = "ml/reports"
DEFAULT_BATCH_SIZE = 1000
SIMILARITY_THRESHOLDS = (0.80, 0.90, 0.95, 0.98)
SIMILARITY_BUCKETS = [
    ("<0.50", 0.0, 0.50),
    ("0.50–0.80", 0.50, 0.80),
    ("0.80–0.90", 0.80, 0.90),
    ("0.90–0.95", 0.90, 0.95),
    ("0.95–0.98", 0.95, 0.98),
    (">=0.98", 0.98, float("inf")),
]


@dataclass
class OverlapMatch:
    split_a: str
    index_a: int
    label_a: int
    label_name_a: str
    split_b: str
    index_b: int
    label_b: int
    label_name_b: str
    same_label: bool
    text_snippet: str


@dataclass
class SplitOverlapResult:
    split_a: str
    split_b: str
    total_a: int
    total_b: int
    raw_exact_matching_pairs: int
    raw_unique_a: int
    raw_unique_b: int
    normalized_exact_matching_pairs: int
    normalized_unique_a: int
    normalized_unique_b: int
    representative_normalized_matches: list[OverlapMatch]


@dataclass
class SuspiciousSimilarityPair:
    query_split: str
    query_index: int
    query_label: int
    query_label_name: str
    train_index: int
    train_label: int
    train_label_name: str
    similarity: float
    same_label: bool
    query_snippet: str
    train_snippet: str


def compute_exact_overlap(
    split_a_data: Sequence[Mapping[str, Any]],
    split_b_data: Sequence[Mapping[str, Any]],
    name_a: str,
    name_b: str,
    label_names: list[str],
    max_representative: int = 10,
    snippet_len: int = 250,
) -> SplitOverlapResult:
    """Compute raw and normalized exact text overlaps between two splits."""
    # 1. Raw text mapping
    raw_map_a: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(split_a_data):
        raw_map_a[str(item["text"])].append(idx)

    raw_pairs_count = 0
    raw_unique_a_affected: set[int] = set()
    raw_unique_b_affected: set[int] = set()
    for idx_b, item in enumerate(split_b_data):
        text_b = str(item["text"])
        if text_b in raw_map_a:
            for idx_a in raw_map_a[text_b]:
                raw_pairs_count += 1
                raw_unique_a_affected.add(idx_a)
                raw_unique_b_affected.add(idx_b)

    # 2. Normalized text mapping
    norm_map_a: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(split_a_data):
        norm_text = normalize_text(str(item["text"]))
        norm_map_a[norm_text].append(idx)

    norm_pairs_count = 0
    norm_unique_a_affected: set[int] = set()
    norm_unique_b_affected: set[int] = set()
    representative_matches: list[OverlapMatch] = []

    for idx_b, item in enumerate(split_b_data):
        raw_b = str(item["text"])
        norm_b = normalize_text(raw_b)
        lbl_b = item["label"]
        if norm_b in norm_map_a:
            for idx_a in norm_map_a[norm_b]:
                norm_pairs_count += 1
                norm_unique_a_affected.add(idx_a)
                norm_unique_b_affected.add(idx_b)
                if len(representative_matches) < max_representative:
                    lbl_a = split_a_data[idx_a]["label"]
                    snippet = " ".join(raw_b.split())
                    if len(snippet) > snippet_len:
                        snippet = snippet[: snippet_len - 1] + "…"
                    representative_matches.append(
                        OverlapMatch(
                            split_a=name_a,
                            index_a=idx_a,
                            label_a=lbl_a,
                            label_name_a=label_names[lbl_a],
                            split_b=name_b,
                            index_b=idx_b,
                            label_b=lbl_b,
                            label_name_b=label_names[lbl_b],
                            same_label=(lbl_a == lbl_b),
                            text_snippet=snippet,
                        )
                    )

    return SplitOverlapResult(
        split_a=name_a,
        split_b=name_b,
        total_a=len(split_a_data),
        total_b=len(split_b_data),
        raw_exact_matching_pairs=raw_pairs_count,
        raw_unique_a=len(raw_unique_a_affected),
        raw_unique_b=len(raw_unique_b_affected),
        normalized_exact_matching_pairs=norm_pairs_count,
        normalized_unique_a=len(norm_unique_a_affected),
        normalized_unique_b=len(norm_unique_b_affected),
        representative_normalized_matches=representative_matches,
    )


def compute_batched_nearest_neighbors(
    vectorizer: TfidfVectorizer,
    train_matrix: Any,
    query_texts: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute maximum cosine similarity and matching train index for query texts in batches.

    Returns
    -------
    max_similarities : np.ndarray
        Array of shape (len(query_texts),) containing max cosine similarity to training set.
    best_train_indices : np.ndarray
        Array of shape (len(query_texts),) containing index in train_matrix of best match.
    """
    total_queries = len(query_texts)
    max_similarities = np.zeros(total_queries, dtype=np.float32)
    best_train_indices = np.zeros(total_queries, dtype=np.int64)

    # Transpose train matrix once: shape (vocab, num_train)
    train_t = train_matrix.T.tocsc()

    for start_idx in range(0, total_queries, batch_size):
        end_idx = min(start_idx + batch_size, total_queries)
        batch_slice = query_texts[start_idx:end_idx]

        # Transform batch to sparse TF-IDF: shape (batch_len, vocab)
        q_batch = vectorizer.transform(batch_slice)

        # Sparse matrix multiplication: shape (batch_len, num_train)
        sim_batch = q_batch.dot(train_t)

        # Extract row-wise maxima without densifying the full matrix
        # sim_batch is CSR
        for i in range(sim_batch.shape[0]):
            row = sim_batch.getrow(i)
            global_idx = start_idx + i
            if row.nnz > 0:
                best_arg = row.data.argmax()
                max_similarities[global_idx] = float(row.data[best_arg])
                best_train_indices[global_idx] = int(row.indices[best_arg])
            else:
                max_similarities[global_idx] = 0.0
                best_train_indices[global_idx] = -1

    return max_similarities, best_train_indices


def summarize_similarity_distribution(
    similarities: np.ndarray,
) -> dict[str, float]:
    """Summarize distribution percentiles and diagnostic threshold proportions."""
    sims = np.asarray(similarities, dtype=np.float32)
    summary: dict[str, float] = {
        "min": float(np.min(sims)),
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "p90": float(np.percentile(sims, 90)),
        "p95": float(np.percentile(sims, 95)),
        "p99": float(np.percentile(sims, 99)),
        "max": float(np.max(sims)),
    }
    total = len(sims)
    for thresh in SIMILARITY_THRESHOLDS:
        count = int(np.sum(sims >= thresh))
        summary[f"count_ge_{thresh:.2f}"] = count
        summary[f"prop_ge_{thresh:.2f}"] = float(count / total) if total else 0.0
    return summary


def assign_similarity_buckets(
    similarities: np.ndarray,
) -> dict[str, Any]:
    """Group similarity scores into reusable standard buckets."""
    sims = np.asarray(similarities, dtype=np.float32)
    total = len(sims)
    bucket_counts: dict[str, int] = {}
    bucket_proportions: dict[str, float] = {}

    for name, low, high in SIMILARITY_BUCKETS:
        if high == float("inf"):
            mask = sims >= low
        else:
            mask = (sims >= low) & (sims < high)
        count = int(np.sum(mask))
        bucket_counts[name] = count
        bucket_proportions[name] = float(count / total) if total else 0.0

    return {
        "total": total,
        "counts": bucket_counts,
        "proportions": bucket_proportions,
    }


def find_top_suspicious_pairs(
    query_split_name: str,
    query_data: Sequence[Mapping[str, Any]],
    train_data: Sequence[Mapping[str, Any]],
    label_names: list[str],
    similarities: np.ndarray,
    best_train_indices: np.ndarray,
    top_n: int = 100,
    snippet_len: int = 250,
) -> list[SuspiciousSimilarityPair]:
    """Extract the top N highest-similarity query pairs against training set."""
    # Sort indices by similarity descending
    top_indices = np.argsort(-similarities)[:top_n]
    pairs: list[SuspiciousSimilarityPair] = []

    for q_idx in top_indices:
        sim = float(similarities[q_idx])
        t_idx = int(best_train_indices[q_idx])
        if t_idx < 0:
            continue

        q_item = query_data[int(q_idx)]
        t_item = train_data[t_idx]

        q_label = int(q_item["label"])
        t_label = int(t_item["label"])

        q_snippet = " ".join(str(q_item["text"]).split())
        if len(q_snippet) > snippet_len:
            q_snippet = q_snippet[: snippet_len - 1] + "…"

        t_snippet = " ".join(str(t_item["text"]).split())
        if len(t_snippet) > snippet_len:
            t_snippet = t_snippet[: snippet_len - 1] + "…"

        pairs.append(
            SuspiciousSimilarityPair(
                query_split=query_split_name,
                query_index=int(q_idx),
                query_label=q_label,
                query_label_name=label_names[q_label],
                train_index=t_idx,
                train_label=t_label,
                train_label_name=label_names[t_label],
                similarity=sim,
                same_label=(q_label == t_label),
                query_snippet=q_snippet,
                train_snippet=t_snippet,
            )
        )

    return pairs


def render_overlap_markdown(results: list[SplitOverlapResult]) -> str:
    """Render split overlap summary as Markdown."""
    lines = [
        "# Split Overlap & Leakage Audit",
        "",
        "> **Note:** Exact-text duplication has been checked, but LexGLUE does not expose "
        "the original LEDGAR source/contract identifier. Split independence cannot be "
        "independently verified at the contract level from this representation.",
        "",
        "## Summary of Overlap Checks",
        "",
        "| Split Pair | Split A Size | Split B Size | Raw Matches | Normalized Matches | Unique A Affected | Unique B Affected |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r.split_a} ↔ {r.split_b} | {r.total_a:,} | {r.total_b:,} | "
            f"{r.raw_exact_matching_pairs:,} | {r.normalized_exact_matching_pairs:,} | "
            f"{r.normalized_unique_a:,} | {r.normalized_unique_b:,} |"
        )

    lines.append("")
    lines.append("## Representative Normalized Exact Matches")
    has_matches = False
    for r in results:
        if r.representative_normalized_matches:
            has_matches = True
            lines.append(f"\n### {r.split_a} ↔ {r.split_b} (Top {len(r.representative_normalized_matches)})")
            for m in r.representative_normalized_matches:
                same_str = "YES" if m.same_label else f"MISMATCH ({m.label_name_a} vs {m.label_name_b})"
                lines.append(
                    f"- **{m.split_a}[{m.index_a}] ({m.label_name_a})** ↔ "
                    f"**{m.split_b}[{m.index_b}] ({m.label_name_b})** [Same Label: {same_str}]"
                )
                lines.append(f"  > `{m.text_snippet}`")
    if not has_matches:
        lines.append("No exact normalized matches were detected between any pair of splits.")

    return "\n".join(lines) + "\n"


def render_similarity_markdown(
    val_summary: dict[str, float],
    test_summary: dict[str, float],
    val_buckets: dict[str, Any],
    test_buckets: dict[str, Any],
) -> str:
    """Render TF-IDF similarity distribution and buckets as Markdown."""
    lines = [
        "# TF-IDF Near-Duplicate Diagnostic Audit",
        "",
        "TF-IDF representation fit on **training split only**.",
        "`TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)`",
        "",
        "## Cosine Similarity to Training Set Statistics",
        "",
        "| Metric | Validation → Train | Test → Train |",
        "| :--- | ---: | ---: |",
        f"| Minimum | {val_summary['min']:.4f} | {test_summary['min']:.4f} |",
        f"| Mean | {val_summary['mean']:.4f} | {test_summary['mean']:.4f} |",
        f"| Median (P50) | {val_summary['median']:.4f} | {test_summary['median']:.4f} |",
        f"| P90 | {val_summary['p90']:.4f} | {test_summary['p90']:.4f} |",
        f"| P95 | {val_summary['p95']:.4f} | {test_summary['p95']:.4f} |",
        f"| P99 | {val_summary['p99']:.4f} | {test_summary['p99']:.4f} |",
        f"| Maximum | {val_summary['max']:.4f} | {test_summary['max']:.4f} |",
        "",
        "## Diagnostic Threshold Proportions",
        "",
        "| Threshold | Validation Count | Val Prop (%) | Test Count | Test Prop (%) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for t in SIMILARITY_THRESHOLDS:
        t_key = f"{t:.2f}"
        lines.append(
            f"| ≥ {t_key} | {val_summary[f'count_ge_{t_key}']:,} | "
            f"{val_summary[f'prop_ge_{t_key}'] * 100:.2f}% | "
            f"{test_summary[f'count_ge_{t_key}']:,} | "
            f"{test_summary[f'prop_ge_{t_key}'] * 100:.2f}% |"
        )

    lines.append("")
    lines.append("## Stratification Buckets (For Model Evaluation)")
    lines.append("")
    lines.append("| Bucket | Validation Count | Val % | Test Count | Test % |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for name, _, _ in SIMILARITY_BUCKETS:
        lines.append(
            f"| {name} | {val_buckets['counts'][name]:,} | "
            f"{val_buckets['proportions'][name] * 100:.2f}% | "
            f"{test_buckets['counts'][name]:,} | "
            f"{test_buckets['proportions'][name] * 100:.2f}% |"
        )

    return "\n".join(lines) + "\n"


def run_leakage_audit(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run full leakage audit (exact overlap + TF-IDF near-duplicate similarity)."""
    from datasets import load_dataset

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {DATASET_NAME}:{DATASET_CONFIG}...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    label_feature = dataset["train"].features["label"]
    label_names = getattr(label_feature, "names", None) or [str(i) for i in range(100)]

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    print("Checking exact raw and normalized overlaps...")
    overlap_train_val = compute_exact_overlap(
        train_data, val_data, "train", "validation", label_names
    )
    overlap_train_test = compute_exact_overlap(
        train_data, test_data, "train", "test", label_names
    )
    overlap_val_test = compute_exact_overlap(
        val_data, test_data, "validation", "test", label_names
    )

    overlap_results = [overlap_train_val, overlap_train_test, overlap_val_test]

    # Export overlap report
    overlap_json_data = [asdict(r) for r in overlap_results]
    with open(out_path / "exact_overlap_audit.json", "w", encoding="utf-8") as f:
        json.dump(overlap_json_data, f, indent=2)

    with open(out_path / "exact_overlap_audit.md", "w", encoding="utf-8") as f:
        f.write(render_overlap_markdown(overlap_results))

    print("Fitting TF-IDF vectorizer on training split only...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_texts = [str(item["text"]) for item in train_data]
    train_matrix = vectorizer.fit_transform(train_texts)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,} features.")

    print(f"Computing nearest-neighbor similarity for validation set (batch size {batch_size})...")
    val_texts = [str(item["text"]) for item in val_data]
    val_sims, val_train_indices = compute_batched_nearest_neighbors(
        vectorizer, train_matrix, val_texts, batch_size=batch_size
    )

    print(f"Computing nearest-neighbor similarity for test set (batch size {batch_size})...")
    test_texts = [str(item["text"]) for item in test_data]
    test_sims, test_train_indices = compute_batched_nearest_neighbors(
        vectorizer, train_matrix, test_texts, batch_size=batch_size
    )

    val_summary = summarize_similarity_distribution(val_sims)
    test_summary = summarize_similarity_distribution(test_sims)

    val_buckets = assign_similarity_buckets(val_sims)
    test_buckets = assign_similarity_buckets(test_sims)

    # Top 100 suspicious pairs
    print("Extracting top suspicious near-duplicate pairs...")
    top_val_pairs = find_top_suspicious_pairs(
        "validation", val_data, train_data, label_names, val_sims, val_train_indices, top_n=100
    )
    top_test_pairs = find_top_suspicious_pairs(
        "test", test_data, train_data, label_names, test_sims, test_train_indices, top_n=100
    )

    # Export machine-readable artifacts
    sim_results = {
        "vectorizer_config": {
            "lowercase": True,
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
            "dtype": "float32",
            "vocab_size": len(vectorizer.vocabulary_),
        },
        "validation_distribution": val_summary,
        "test_distribution": test_summary,
        "validation_buckets": val_buckets,
        "test_buckets": test_buckets,
    }
    with open(out_path / "tfidf_similarity_audit.json", "w", encoding="utf-8") as f:
        json.dump(sim_results, f, indent=2)

    with open(out_path / "top_100_validation_pairs.json", "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in top_val_pairs], f, indent=2)

    with open(out_path / "top_100_test_pairs.json", "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in top_test_pairs], f, indent=2)

    # Save compact similarity arrays for future evaluation stratification
    np.save(out_path / "val_similarity_to_train.npy", val_sims)
    np.save(out_path / "test_similarity_to_train.npy", test_sims)

    # Export Markdown reports
    with open(out_path / "tfidf_similarity_audit.md", "w", encoding="utf-8") as f:
        f.write(render_similarity_markdown(val_summary, test_summary, val_buckets, test_buckets))

    print("Leakage audit complete! Artifacts saved to:", out_path.resolve())
    return {
        "overlap": overlap_json_data,
        "similarity": sim_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit leakage and near-duplicate similarity.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write audit artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for sparse matrix similarity (default: {DEFAULT_BATCH_SIZE})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    run_leakage_audit(output_dir=args.output_dir, batch_size=args.batch_size)
