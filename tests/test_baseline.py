"""Synthetic tests for the TF-IDF baseline; these never download LEDGAR."""

from __future__ import annotations

import json

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.ipc as ipc

from ml.src.baseline import (
    build_logistic_regression,
    build_metadata,
    build_tfidf,
    evaluate_fitted,
    transform_splits,
)
from ml.src.data_loader import LedgarData, Split, load_from_cache_dir


def _synthetic_data() -> LedgarData:
    return LedgarData(
        splits={
            "train": Split(
                texts=[
                    "payment shall be made promptly",
                    "payment must be received promptly",
                    "confidential information shall remain private",
                    "confidential records must remain private",
                    "termination occurs after notice",
                    "termination follows written notice",
                    "renewal follows the effective date",
                    "renewal extends the agreement term",
                ],
                labels=[0, 0, 1, 1, 2, 2, 3, 3],
            ),
            "validation": Split(
                texts=[
                    "payment shall be made",
                    "confidential information stays private",
                    "termination follows notice unseenvalidationtoken",
                    "renewal extends the contract",
                ],
                labels=[0, 1, 2, 3],
            ),
            "test": Split(texts=["not touched"], labels=[0]),
        },
        label_names=["Payment", "Confidentiality", "Termination", "Renewal"],
        loader="synthetic",
    )


def test_tfidf_fits_train_and_transforms_validation() -> None:
    data = _synthetic_data()
    vectorizer = build_tfidf()
    matrices = transform_splits(data, vectorizer)

    assert matrices["X_train"].shape[0] == 8
    assert matrices["X_validation"].shape[0] == 4
    assert matrices["y_train"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert matrices["y_validation"].tolist() == [0, 1, 2, 3]
    assert "unseenvalidationtoken" not in vectorizer.vocabulary_
    assert vectorizer.get_params()["stop_words"] is None
    assert vectorizer.get_params()["ngram_range"] == (1, 2)
    assert vectorizer.get_params()["dtype"] is np.float32


def test_logistic_pipeline_predictions_and_evaluation() -> None:
    data = _synthetic_data()
    matrices = transform_splits(data, build_tfidf())
    classifier = build_logistic_regression(C=1.0)
    classifier.fit(matrices["X_train"], matrices["y_train"])

    predictions = classifier.predict(matrices["X_validation"])
    assert predictions.shape == (4,)
    result = evaluate_fitted(
        classifier, matrices["X_validation"], matrices["y_validation"], data.label_names
    )
    assert set(result["metrics"]) == {"accuracy", "macro_f1", "weighted_f1"}
    assert len(result["per_class"]) == 4
    assert set(result["top_k"]) == {
        "top_1_accuracy",
        "top_3_accuracy",
    }
    assert len(result["confusion"]["matrix"]) == 4


def test_metadata_configuration_is_json_serializable() -> None:
    metadata = build_metadata(
        _synthetic_data(),
        "tfidf-logreg",
        split="validation",
        configuration={"dtype": np.float32, "ngram_range": (1, 2)},
    )
    serialized = json.dumps(metadata.to_dict())
    assert '"dtype": "float32"' in serialized
    assert '"ngram_range": [1, 2]' in serialized


def test_final_evaluation_refuses_to_overwrite_test_report(tmp_path) -> None:
    from ml.src.baseline import run_final

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "test_evaluation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to evaluate test again"):
        run_final(_synthetic_data(), "logreg", 1.0, None, reports, tmp_path / "models")


def test_cache_loader_reads_only_requested_splits(tmp_path) -> None:
    (tmp_path / "dataset_info.json").write_text(
        json.dumps(
            {
                "features": {
                    "label": {"names": [f"class-{i}" for i in range(100)]}
                },
                "version": {"version_str": "test-revision"},
            }
        ),
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        table = pa.table({"text": [f"{split} text"], "label": [0]})
        with pa.OSFile(str(tmp_path / f"lex_glue-{split}.arrow"), "wb") as sink:
            with ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)

    data = load_from_cache_dir(tmp_path, splits=("train", "validation"))
    assert list(data.splits) == ["train", "validation"]
    assert data.splits["validation"].texts == ["validation text"]
    assert data.version == "test-revision"
