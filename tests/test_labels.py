"""Tests for dataset label audit utilities using synthetic fixtures."""

from __future__ import annotations

from collections import Counter

import pytest

from ml.src.audit_labels import (
    LabelSupportRow,
    analyze_rare_classes,
    build_label_support_table,
    count_labels_by_split,
    extract_label_names_from_feature,
    select_representative_examples,
)


class DummyFeature:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.num_classes = len(names)

    def int2str(self, idx: int) -> str:
        return self.names[idx]


def test_extract_label_names() -> None:
    feature = DummyFeature(["Term", "Termination", "Notices"])
    names = extract_label_names_from_feature(feature)
    assert names == ["Term", "Termination", "Notices"]


def test_count_labels_by_split() -> None:
    dataset = {
        "train": [{"label": 0}, {"label": 1}, {"label": 1}],
        "validation": [{"label": 1}],
        "test": [{"label": 0}],
    }
    counts = count_labels_by_split(dataset)
    assert counts["train"][0] == 1
    assert counts["train"][1] == 2
    assert counts["validation"][1] == 1
    assert counts["test"][0] == 1


def test_build_label_support_table() -> None:
    names = ["CatA", "CatB", "CatC"]
    counts_by_split = {
        "train": Counter({0: 10, 1: 50, 2: 5}),
        "validation": Counter({0: 2, 1: 10, 2: 1}),
        "test": Counter({0: 3, 1: 15, 2: 2}),
    }
    table = build_label_support_table(names, counts_by_split)
    # Sorted descending by total_count:
    # CatB: 50+10+15 = 75
    # CatA: 10+2+3 = 15
    # CatC: 5+1+2 = 8
    assert len(table) == 3
    assert table[0].label_name == "CatB"
    assert table[0].total_count == 75
    assert table[1].label_name == "CatA"
    assert table[1].total_count == 15
    assert table[2].label_name == "CatC"
    assert table[2].total_count == 8


def test_analyze_rare_classes() -> None:
    rows = [
        LabelSupportRow(0, "A", 20, 5, 5, 30),     # <50
        LabelSupportRow(1, "B", 40, 20, 10, 70),   # 50-99
        LabelSupportRow(2, "C", 100, 30, 20, 150), # 100-249
        LabelSupportRow(3, "D", 200, 50, 50, 300), # 250-499
        LabelSupportRow(4, "E", 400, 100, 100, 600),# 500+
    ]
    buckets = analyze_rare_classes(rows)
    bucket_map = {b.bucket_name: b for b in buckets}

    assert bucket_map["<50"].class_count == 1
    assert bucket_map["<50"].total_support == 30

    assert bucket_map["50–99"].class_count == 1
    assert bucket_map["50–99"].total_support == 70

    assert bucket_map["100–249"].class_count == 1
    assert bucket_map["100–249"].total_support == 150

    assert bucket_map["250–499"].class_count == 1
    assert bucket_map["250–499"].total_support == 300

    assert bucket_map["500+"].class_count == 1
    assert bucket_map["500+"].total_support == 600


def test_select_representative_examples() -> None:
    dataset = {
        "train": [
            {"text": "Sample train text 0", "label": 0},
            {"text": "Sample train text 1", "label": 1},
        ],
        "validation": [
            {"text": "Sample val text 0", "label": 0},
        ],
    }
    examples = select_representative_examples(dataset, ["ClassA", "ClassB"], examples_per_class=2)
    assert len(examples[0]) == 2
    assert examples[0][0].split == "train"
    assert examples[0][1].split == "validation"
    assert len(examples[1]) == 1
    assert examples[1][0].split == "train"
