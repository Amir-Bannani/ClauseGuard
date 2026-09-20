"""Deterministically inspect the LexGLUE LEDGAR dataset without modifying it.

Run from the repository root:
    python ml/src/inspect_dataset.py

Hugging Face Datasets manages its own local cache, so subsequent runs reuse
downloaded data when it is still available locally.
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any


DATASET_PATH = "coastalcph/lex_glue"
DATASET_CONFIG = "ledgar"
SEED = 20260920
WORD_PATTERN = re.compile(r"\S+")
PERCENTILES = (50, 90, 95, 99)


def percentile(values: Sequence[int], percentage: int) -> float:
    """Return a deterministic, linearly interpolated percentile."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def describe_lengths(values: Sequence[int]) -> dict[str, float | int]:
    """Summarize lengths without requiring a tokenizer or ML dependency."""
    if not values:
        raise ValueError("Cannot describe an empty sequence.")
    result: dict[str, float | int] = {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }
    result.update({f"p{p}": percentile(values, p) for p in PERCENTILES})
    return result


def format_number(value: float | int) -> str:
    return f"{value:,.1f}" if isinstance(value, float) else f"{value:,}"


def shorten(value: str, limit: int = 280) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


def find_column(features: Mapping[str, Any], preferred_names: Sequence[str], kind: str) -> str:
    for name in preferred_names:
        if name in features:
            return name
    available = ", ".join(features)
    raise RuntimeError(f"Could not identify the {kind} column. Available columns: {available}")


def label_name(feature: Any, raw_label: Any) -> str | None:
    """Resolve ClassLabel IDs when the dataset exposes label names."""
    int2str = getattr(feature, "int2str", None)
    if callable(int2str) and isinstance(raw_label, int):
        return int2str(raw_label)
    names = getattr(feature, "names", None)
    if isinstance(names, list) and isinstance(raw_label, int) and 0 <= raw_label < len(names):
        return names[raw_label]
    return None


def print_example(example: Mapping[str, Any], text_column: str, label_column: str, label_feature: Any) -> None:
    raw_label = example[label_column]
    readable_label = label_name(label_feature, raw_label)
    print(f"  raw label: {raw_label!r}")
    print(f"  label name: {readable_label if readable_label is not None else 'not available'}")
    print(f"  text: {shorten(str(example[text_column]))}")


def load_ledgar() -> Any:
    """Import lazily so helper functions can be tested without downloading data."""
    from datasets import load_dataset

    return load_dataset(DATASET_PATH, DATASET_CONFIG)


def inspect_dataset(seed: int = SEED, examples_per_split: int = 2, examples_per_category: int = 1) -> None:
    dataset = load_ledgar()
    split_names = list(dataset.keys())
    if not split_names:
        raise RuntimeError("The loaded dataset contains no splits.")

    reference_features = dataset[split_names[0]].features
    text_column = find_column(reference_features, ("text", "provision", "sentence"), "text")
    label_column = find_column(reference_features, ("label", "labels"), "label")
    label_feature = reference_features[label_column]

    print("# LexGLUE LEDGAR dataset inspection")
    print(f"Dataset: {DATASET_PATH!r}, config: {DATASET_CONFIG!r}")
    print(f"Sampling seed: {seed}")
    print(f"Text column selected after schema inspection: {text_column!r}")
    print(f"Label column selected after schema inspection: {label_column!r}\n")

    print("## Splits and schema")
    for split_name in split_names:
        split = dataset[split_name]
        print(f"- {split_name}: {len(split):,} examples")
        print(f"  columns: {list(split.column_names)!r}")
        print(f"  features: {split.features!r}")

    random_generator = random.Random(seed)
    print("\n## Representative raw examples")
    for split_name in split_names:
        split = dataset[split_name]
        count = min(examples_per_split, len(split))
        indices = sorted(random_generator.sample(range(len(split)), count))
        print(f"\n### {split_name}")
        for index in indices:
            print(f"Example {index}:")
            print_example(split[index], text_column, label_column, label_feature)

    labels: Counter[Any] = Counter()
    texts_by_split: dict[str, set[str]] = {}
    text_occurrences: Counter[str] = Counter()
    examples_by_label: dict[Any, list[tuple[str, int, Mapping[str, Any]]]] = defaultdict(list)
    all_texts: list[str] = []
    for split_name in split_names:
        split_texts: set[str] = set()
        for index, example in enumerate(dataset[split_name]):
            text = str(example[text_column])
            raw_label = example[label_column]
            labels[raw_label] += 1
            text_occurrences[text] += 1
            split_texts.add(text)
            all_texts.append(text)
            if len(examples_by_label[raw_label]) < examples_per_category:
                examples_by_label[raw_label].append((split_name, index, example))
        texts_by_split[split_name] = split_texts

    category_names = {
        raw_label: label_name(label_feature, raw_label) or str(raw_label) for raw_label in labels
    }
    counts = list(labels.values())
    print("\n## Label analysis")
    print(f"Unique categories observed: {len(labels):,}")
    print(f"Minimum class size: {min(counts):,}")
    print(f"Maximum class size: {max(counts):,}")
    print(f"Median class size: {statistics.median(counts):,.1f}")
    print("\nAll category names (label ID -> category):")
    for raw_label in sorted(labels, key=lambda value: (category_names[value].casefold(), value)):
        print(f"- {raw_label!r} -> {category_names[raw_label]}")
    print("\nSorted class distribution (ascending by count):")
    print("| Raw label | Category | Examples |")
    print("| ---: | --- | ---: |")
    for raw_label, count in sorted(labels.items(), key=lambda item: (item[1], category_names[item[0]].casefold())):
        print(f"| {raw_label!r} | {category_names[raw_label]} | {count:,} |")

    character_lengths = [len(text) for text in all_texts]
    word_lengths = [len(WORD_PATTERN.findall(text)) for text in all_texts]
    print("\n## Text length analysis")
    for metric, values in (("Characters", character_lengths), ("Whitespace-delimited word approximation", word_lengths)):
        summary = describe_lengths(values)
        print(f"\n{metric}:")
        print(
            "  " + ", ".join(f"{key}={format_number(value)}" for key, value in summary.items())
        )
    ranked_indices = sorted(range(len(all_texts)), key=lambda index: (character_lengths[index], all_texts[index]))
    print("\nVery short examples (by character length):")
    for index in ranked_indices[:3]:
        print(f"- {character_lengths[index]:,} chars / {word_lengths[index]:,} words: {shorten(all_texts[index])}")
    print("\nVery long examples (by character length):")
    for index in ranked_indices[-3:]:
        print(f"- {character_lengths[index]:,} chars / {word_lengths[index]:,} words: {shorten(all_texts[index])}")

    duplicate_groups = [(text, count) for text, count in text_occurrences.items() if count > 1]
    duplicate_groups.sort(key=lambda item: (-item[1], item[0]))
    total_examples = len(all_texts)
    print("\n## Exact duplicate-text analysis")
    print(f"Total examples: {total_examples:,}")
    print(f"Unique texts: {len(text_occurrences):,}")
    print(f"Duplicate occurrences beyond the first: {total_examples - len(text_occurrences):,}")
    print(f"Distinct text values that are duplicated: {len(duplicate_groups):,}")
    print("Most duplicated texts:")
    for text, count in duplicate_groups[:5]:
        print(f"- {count:,} occurrences: {shorten(text)}")

    print("\n## Exact text overlap between provided splits")
    for first, second in combinations(split_names, 2):
        overlap = texts_by_split[first] & texts_by_split[second]
        print(f"- {first} intersect {second}: {len(overlap):,} unique identical texts")
        for text in sorted(overlap)[:3]:
            print(f"  - {shorten(text)}")

    print("\n## Representative examples by category")
    for raw_label in sorted(labels, key=lambda value: (category_names[value].casefold(), value)):
        print(f"\n### {raw_label!r} — {category_names[raw_label]}")
        for split_name, index, example in examples_by_label[raw_label]:
            print(f"{split_name}[{index}]:")
            print_example(example, text_column, label_column, label_feature)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for split-level samples.")
    parser.add_argument("--examples-per-split", type=int, default=2)
    parser.add_argument("--examples-per-category", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    # The dataset contains Unicode legal text; make direct Windows-console runs
    # behave the same as redirected UTF-8 output where supported.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    arguments = parse_arguments()
    inspect_dataset(
        seed=arguments.seed,
        examples_per_split=arguments.examples_per_split,
        examples_per_category=arguments.examples_per_category,
    )
