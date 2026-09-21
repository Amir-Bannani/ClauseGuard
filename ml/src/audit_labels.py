"""Audit label support, distribution, rare classes, and representative examples for LexGLUE LEDGAR.

Run from repository root:
    python ml/src/audit_labels.py [--output-dir ml/reports]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DATASET_NAME = "coastalcph/lex_glue"
DATASET_CONFIG = "ledgar"
DEFAULT_OUTPUT_DIR = "ml/reports"
DEFAULT_EXAMPLES_PER_CLASS = 3
DEFAULT_SEED = 20260921

SUPPORT_BUCKET_DEFINITIONS = [
    ("<50", 0, 49),
    ("50–99", 50, 99),
    ("100–249", 100, 249),
    ("250–499", 250, 499),
    ("500+", 500, float("inf")),
]


@dataclass
class LabelSupportRow:
    label_id: int
    label_name: str
    train_count: int
    validation_count: int
    test_count: int
    total_count: int


@dataclass
class RepresentativeExample:
    split: str
    dataset_index: int
    label_id: int
    label_name: str
    text_snippet: str
    character_length: int


@dataclass
class RareClassBucket:
    bucket_name: str
    min_count: int
    max_count: float
    class_count: int
    train_support: int
    validation_support: int
    test_support: int
    total_support: int
    classes: list[dict[str, Any]]


def extract_label_names_from_feature(label_feature: Any) -> list[str]:
    """Extract list of label names from a ClassLabel feature."""
    names = getattr(label_feature, "names", None)
    if isinstance(names, list) and names:
        return list(names)
    num_classes = getattr(label_feature, "num_classes", None)
    int2str = getattr(label_feature, "int2str", None)
    if callable(int2str) and isinstance(num_classes, int):
        return [int2str(i) for i in range(num_classes)]
    raise ValueError(f"Unable to extract label names from feature {label_feature!r}")


def count_labels_by_split(
    split_data: Mapping[str, Sequence[Mapping[str, Any]] | Any],
    label_column: str = "label",
) -> dict[str, Counter[int]]:
    """Count occurrences of each integer label per split."""
    counts_by_split: dict[str, Counter[int]] = {}
    for split_name, split in split_data.items():
        counts = Counter(example[label_column] for example in split)
        counts_by_split[split_name] = counts
    return counts_by_split


def build_label_support_table(
    label_names: list[str],
    counts_by_split: dict[str, Counter[int]],
) -> list[LabelSupportRow]:
    """Build support table for all classes, sorted by total_count descending."""
    train_counts = counts_by_split.get("train", Counter())
    val_counts = counts_by_split.get("validation", Counter())
    test_counts = counts_by_split.get("test", Counter())

    table: list[LabelSupportRow] = []
    for label_id, name in enumerate(label_names):
        train_c = train_counts.get(label_id, 0)
        val_c = val_counts.get(label_id, 0)
        test_c = test_counts.get(label_id, 0)
        total_c = train_c + val_c + test_c
        table.append(
            LabelSupportRow(
                label_id=label_id,
                label_name=name,
                train_count=train_c,
                validation_count=val_c,
                test_count=test_c,
                total_count=total_c,
            )
        )

    # Sort descending by total_count, secondary by label_name ascending
    table.sort(key=lambda row: (-row.total_count, row.label_name.casefold()))
    return table


def analyze_rare_classes(
    support_table: list[LabelSupportRow],
) -> list[RareClassBucket]:
    """Group classes into predefined support buckets."""
    buckets: list[RareClassBucket] = []

    for name, low, high in SUPPORT_BUCKET_DEFINITIONS:
        matching_rows = [row for row in support_table if low <= row.total_count <= high]
        bucket = RareClassBucket(
            bucket_name=name,
            min_count=low,
            max_count=high,
            class_count=len(matching_rows),
            train_support=sum(r.train_count for r in matching_rows),
            validation_support=sum(r.validation_count for r in matching_rows),
            test_support=sum(r.test_count for r in matching_rows),
            total_support=sum(r.total_count for r in matching_rows),
            classes=[
                {
                    "label_id": r.label_id,
                    "label_name": r.label_name,
                    "total_count": r.total_count,
                    "train_count": r.train_count,
                    "validation_count": r.validation_count,
                    "test_count": r.test_count,
                }
                for r in matching_rows
            ],
        )
        buckets.append(bucket)

    return buckets


def select_representative_examples(
    split_data: Mapping[str, Sequence[Mapping[str, Any]] | Any],
    label_names: list[str],
    examples_per_class: int = DEFAULT_EXAMPLES_PER_CLASS,
    text_column: str = "text",
    label_column: str = "label",
    snippet_len: int = 300,
) -> dict[int, list[RepresentativeExample]]:
    """Select 2-3 deterministic representative examples for every class across splits.

    Preference is given to the training split first, then validation, then test,
    so that examples are deterministic and reproducible without stochastic sampling.
    """
    selected: dict[int, list[RepresentativeExample]] = defaultdict(list)
    desired_splits = ["train", "validation", "test"]

    for split_name in desired_splits:
        if split_name not in split_data:
            continue
        split = split_data[split_name]
        for idx, item in enumerate(split):
            lbl = item[label_column]
            if len(selected[lbl]) < examples_per_class:
                full_text = str(item[text_column])
                snippet = " ".join(full_text.split())
                if len(snippet) > snippet_len:
                    snippet = snippet[: snippet_len - 1] + "…"
                selected[lbl].append(
                    RepresentativeExample(
                        split=split_name,
                        dataset_index=idx,
                        label_id=lbl,
                        label_name=label_names[lbl],
                        text_snippet=snippet,
                        character_length=len(full_text),
                    )
                )

    return dict(selected)


def render_support_table_markdown(table: list[LabelSupportRow]) -> str:
    """Render full 100-class support table as GitHub Flavored Markdown."""
    lines = [
        "# LexGLUE LEDGAR Label Support Table",
        "",
        "Sorted primarily by total support count descending.",
        "",
        "| Rank | Label ID | Label Name | Train | Validation | Test | Total Support |",
        "| ---: | ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(table, 1):
        lines.append(
            f"| {rank} | {row.label_id} | {row.label_name} | {row.train_count:,} | "
            f"{row.validation_count:,} | {row.test_count:,} | {row.total_count:,} |"
        )
    return "\n".join(lines) + "\n"


def render_rare_class_markdown(buckets: list[RareClassBucket]) -> str:
    """Render rare-class bucket summary as Markdown."""
    lines = [
        "# Rare Class Support Analysis",
        "",
        "| Bucket | Classes | Train Support | Val Support | Test Support | Total Support | % of Total |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    grand_total = sum(b.total_support for b in buckets)
    for b in buckets:
        pct = (b.total_support / grand_total * 100) if grand_total else 0.0
        lines.append(
            f"| {b.bucket_name} | {b.class_count} | {b.train_support:,} | "
            f"{b.validation_support:,} | {b.test_support:,} | {b.total_support:,} | {pct:.2f}% |"
        )

    lines.append("")
    lines.append("## Detailed Classes in Rare Buckets (<100 total examples)")
    for b in buckets:
        if b.max_count < 100 and b.classes:
            lines.append(f"\n### Bucket: {b.bucket_name} ({b.class_count} classes)")
            for cls in b.classes:
                lines.append(
                    f"- **{cls['label_name']}** (ID: {cls['label_id']}): "
                    f"total={cls['total_count']:,} (train={cls['train_count']:,}, "
                    f"val={cls['validation_count']:,}, test={cls['test_count']:,})"
                )

    return "\n".join(lines) + "\n"


def run_label_audit(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Execute the complete label audit against the cached LEDGAR dataset."""
    from datasets import load_dataset

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {DATASET_NAME}:{DATASET_CONFIG}...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # 1. Label names extraction directly from ClassLabel metadata
    train_features = dataset["train"].features
    label_feature = train_features["label"]
    label_names = extract_label_names_from_feature(label_feature)
    print(f"Discovered {len(label_names)} classes from dataset metadata.")

    # 2. Count support per split
    counts_by_split = count_labels_by_split(dataset, label_column="label")

    # 3. Build complete support table
    support_table = build_label_support_table(label_names, counts_by_split)

    # 4. Rare class analysis
    rare_buckets = analyze_rare_classes(support_table)

    # 5. Representative examples
    representative_examples = select_representative_examples(
        dataset, label_names, examples_per_class=DEFAULT_EXAMPLES_PER_CLASS
    )

    # Export machine-readable outputs
    support_table_data = [asdict(r) for r in support_table]
    with open(out_path / "label_support_table.json", "w", encoding="utf-8") as f:
        json.dump(support_table_data, f, indent=2)

    with open(out_path / "label_names.json", "w", encoding="utf-8") as f:
        json.dump({"num_classes": len(label_names), "label_names": label_names}, f, indent=2)

    rare_buckets_data = [asdict(b) for b in rare_buckets]
    with open(out_path / "rare_class_analysis.json", "w", encoding="utf-8") as f:
        json.dump(rare_buckets_data, f, indent=2)

    examples_data = {
        str(k): [asdict(ex) for ex in v] for k, v in representative_examples.items()
    }
    with open(out_path / "representative_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples_data, f, indent=2)

    # Export human-readable Markdown reports
    with open(out_path / "label_support_table.md", "w", encoding="utf-8") as f:
        f.write(render_support_table_markdown(support_table))

    with open(out_path / "rare_class_analysis.md", "w", encoding="utf-8") as f:
        f.write(render_rare_class_markdown(rare_buckets))

    summary = {
        "num_classes": len(label_names),
        "total_examples": sum(r.total_count for r in support_table),
        "train_examples": sum(r.train_count for r in support_table),
        "validation_examples": sum(r.validation_count for r in support_table),
        "test_examples": sum(r.test_count for r in support_table),
        "min_class_size": support_table[-1].total_count,
        "min_class_name": support_table[-1].label_name,
        "max_class_size": support_table[0].total_count,
        "max_class_name": support_table[0].label_name,
        "buckets": [
            {
                "bucket": b.bucket_name,
                "class_count": b.class_count,
                "total_support": b.total_support,
            }
            for b in rare_buckets
        ],
    }

    with open(out_path / "label_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Label audit complete! Artifacts saved to:", out_path.resolve())
    print(f"Top class: {summary['max_class_name']} ({summary['max_class_size']:,})")
    print(f"Bottom class: {summary['min_class_name']} ({summary['min_class_size']:,})")
    for b in rare_buckets:
        print(f"  Bucket {b.bucket_name}: {b.class_count} classes, {b.total_support:,} provisions")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LexGLUE LEDGAR labels.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write audit artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    run_label_audit(output_dir=args.output_dir)
