"""Load the LexGLUE LEDGAR dataset for clause-categorization experiments.

Two load paths are provided:

1. A cached-arrow path that reads the locally cached Hugging Face dataset
   files (Arrow IPC streams) with **pyarrow only** — no ``datasets``/``pandas``
   dependency. A complete cache is preferred for deterministic offline reruns.
2. ``datasets.load_dataset`` when no complete local cache exists.

The fallback preserves the dataset's own 100-class ``ClassLabel`` metadata:
label names are read from the ``dataset_info.json`` manifest shipped with the
cache, never hard-coded. Both paths return the same ``LedgarData`` structure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_NAME = "coastalcph/lex_glue"
DATASET_CONFIG = "ledgar"
SPLITS = ("train", "validation", "test")


@dataclass
class Split:
    """One dataset split: aligned texts and integer labels."""

    texts: list[str]
    labels: list[int]


@dataclass
class LedgarData:
    """Loaded LEDGAR data for the baseline experiment.

    Attributes
    ----------
    splits : dict[str, Split]
        Keys are ``train``, ``validation``, ``test``.
    label_names : list[str]
        The 100 ``ClassLabel`` names in label-id order, from dataset metadata.
    loader : str
        ``"datasets"`` when loaded through Hugging Face ``datasets``, or
        ``"cached-arrow"`` when read from the local cache via pyarrow.
    version : str | None
        Dataset version from the cache manifest when available.
    """

    splits: dict[str, Split]
    label_names: list[str]
    loader: str
    dataset_name: str = DATASET_NAME
    dataset_config: str = DATASET_CONFIG
    version: str | None = None

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


def _cache_prefix(name: str) -> str:
    """Convert 'org/name' to the cache directory prefix 'org___name'."""
    return name.replace("/", "___")


def _find_cache_dir(dataset_name: str = DATASET_NAME, config: str = DATASET_CONFIG) -> Path:
    """Locate the HF cache directory holding the Arrow files for this dataset."""
    cache_root = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "datasets"
    base = cache_root / _cache_prefix(dataset_name)
    if not base.exists():
        raise FileNotFoundError(
            f"Hugging Face cache not found at {base}. Populate it with:\n"
            "  python -c \"from datasets import load_dataset; "
            f"load_dataset('{dataset_name}', '{config}')\""
        )
    for version_dir in base.glob(f"{config}/*/*"):
        info_file = version_dir / "dataset_info.json"
        has_splits = all((version_dir / f"lex_glue-{split}.arrow").exists() for split in SPLITS)
        if info_file.exists() and has_splits:
            return version_dir
    raise FileNotFoundError(
        f"No complete cached version of '{dataset_name}' config '{config}' "
        f"found under {base}."
    )


def _read_cache_split(path: Path) -> Split:
    """Read one cached Arrow IPC stream into a Split (pyarrow only)."""
    import pyarrow.ipc as ipc

    with ipc.open_stream(str(path)) as reader:
        table = reader.read_all()
    texts = [str(value) for value in table.column("text").to_pylist()]
    labels = [int(value) for value in table.column("label").to_pylist()]
    if len(texts) != len(labels):
        raise ValueError(f"text/label length mismatch in {path}")
    return Split(texts=texts, labels=labels)


def _read_cache_version(info_path: Path) -> str | None:
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        raw = info.get("version")
        if raw is None:
            return None
        if isinstance(raw, dict):
            return str(raw.get("version_str", raw))
        return str(raw)
    except (OSError, ValueError):
        return None


def load_from_cache_dir(cache_dir: Path, splits: tuple[str, ...] = SPLITS) -> LedgarData:
    """Load LEDGAR directly from a HF-cache version directory (pyarrow only)."""
    cache_dir = Path(cache_dir)
    info_path = cache_dir / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No dataset_info.json in {cache_dir}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    label_feature: Any = None
    features = info.get("features")
    if isinstance(features, dict):
        label_feature = features.get("label")
    if isinstance(label_feature, dict):
        label_names = list(label_feature.get("names", []))
    else:
        label_names = []
    if len(label_names) != 100:
        raise ValueError(
            f"Expected 100 label names from cache metadata, got {len(label_names)}"
        )

    requested_splits = tuple(dict.fromkeys(splits))
    unknown = set(requested_splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown LEDGAR splits: {sorted(unknown)}")
    loaded_splits: dict[str, Split] = {}
    for split in requested_splits:
        path = cache_dir / f"lex_glue-{split}.arrow"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached split file {path}")
        loaded_splits[split] = _read_cache_split(path)

    return LedgarData(
        splits=loaded_splits,
        label_names=label_names,
        loader="cached-arrow",
        version=_read_cache_version(info_path),
    )


def _load_with_datasets(splits: tuple[str, ...] = SPLITS) -> LedgarData:
    """Load LEDGAR through Hugging Face ``datasets`` (requires pandas)."""
    from datasets import load_dataset

    requested_splits = tuple(dict.fromkeys(splits))
    unknown = set(requested_splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown LEDGAR splits: {sorted(unknown)}")
    if "train" not in requested_splits:
        raise ValueError("LEDGAR loading requires the training split for label metadata")
    train_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    label_names = list(train_dataset.features["label"].names)
    loaded_splits: dict[str, Split] = {}
    for split in requested_splits:
        ds_split = train_dataset if split == "train" else load_dataset(
            DATASET_NAME, DATASET_CONFIG, split=split
        )
        texts = [str(example["text"]) for example in ds_split]
        labels = [int(example["label"]) for example in ds_split]
        loaded_splits[split] = Split(texts=texts, labels=labels)
    return LedgarData(
        splits=loaded_splits,
        label_names=label_names,
        loader="datasets",
        version=None,
    )


def load_ledgar(splits: tuple[str, ...] = SPLITS) -> LedgarData:
    """Load LexGLUE LEDGAR from cache when present, otherwise use datasets.

    Preferring the complete local cache makes reruns independent of network
    availability and avoids remote revision changes mid-experiment.
    """
    try:
        cache_dir = _find_cache_dir()
    except FileNotFoundError:
        cache_dir = None
    if cache_dir is not None:
        data = load_from_cache_dir(cache_dir, splits=splits)
        print(
            f"Loaded '{data.dataset_name}' config '{data.dataset_config}' from "
            f"Hugging Face cache via pyarrow: "
            f"{ {s: len(data.splits[s].texts) for s in data.splits} }"
        )
        return data

    try:
        return _load_with_datasets(splits=splits)
    except (ImportError, AttributeError):
        raise RuntimeError(
            "The datasets package could not be imported and no complete LEDGAR "
            "cache was found. Populate the Hugging Face cache before training."
        )
