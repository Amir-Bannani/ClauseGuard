# Evaluation protocol and results

This document defines the LEDGAR split discipline and metrics used by the
TF-IDF baseline. The reusable implementation is under `ml/src/evaluation/`.

## Split discipline

LexGLUE `coastalcph/lex_glue`, configuration `ledgar`, supplies predetermined
chronological splits: 60,000 train, 10,000 validation, and 10,000 test
examples. They have fixed roles:

```text
TRAIN      -> fit the vectorizer and classifier
VALIDATION -> select hyperparameters/models and perform error analysis
TEST       -> evaluate the frozen configuration once
```

The baseline CLI loads train and validation only for tuning and diagnostics.
The final command loads train and test only after the configuration is
selected. TF-IDF is fit on train only. Similarity-to-training values are
diagnostic and never model inputs or selection criteria. Test results are
reported without subsequent tuning.

The package exposes no source-contract identifier, so grouping by contract
cannot be verified. The dataset's chronological split is retained as supplied.

## Metrics

The task has 100 single-label classes with strong imbalance (largest class:
Governing Laws, 4,243 examples; smallest: Books, 25). Macro-F1 is computed over
the fixed set of all 100 class IDs, including a zero score for a class with no
validation support and no predictions.

| Metric | Role |
| --- | --- |
| Macro-F1 | Primary selection criterion; averages F1 equally over all 100 labels. |
| Accuracy | Overall top-1 correctness. |
| Weighted-F1 | F1 averaged by class support. |
| Per-class precision, recall, F1, support | Shows class-specific strengths and weaknesses. |
| Top-1 / top-3 / top-5 | True class rank from probabilities or real-valued class scores. |
| Confusion matrix and top confused pairs | Full 100×100 matrix and largest off-diagonal errors. |

For single-label multiclass data, micro-F1 equals accuracy and is omitted as a
redundant metric. Logistic Regression top-k uses `predict_proba`; LinearSVC
top-k uses `decision_function` and is a ranking measure, not a calibrated
probability.

The implementation in `metrics.py` uses sklearn metric functions and accepts a
fixed label list for macro-F1 and per-class reports. This ensures a class with
zero support does not silently disappear from the primary metric.

## Selected baseline results

All selection below used validation only. The highest tested macro-F1 was
LinearSVC (`C=3`, no class weighting): accuracy 0.8828, macro-F1 0.8194,
weighted-F1 0.8795, top-3 0.9535, and top-5 0.9660. It was selected over the
strongest Logistic Regression configuration (`C=10`, no class weighting;
macro-F1 0.8068). The full C grids and comparison are in
[ml-pipeline.md](ml-pipeline.md).

The selected configuration was then evaluated once on the untouched test
split: accuracy 0.8797, macro-F1 0.8299, weighted-F1 0.8757, top-3 0.9506,
and top-5 0.9621. Test metrics did not change the configuration.

## Error analysis and similarity diagnostics

Error analysis was performed on validation for Logistic Regression `C=10`,
unweighted, before comparing LinearSVC. The largest confusion was Applicable
Laws → Governing Laws (50 cases); other recurring pairs included Defined
Terms/Definitions, Tax Withholdings/Withholdings, and No Waivers/Waivers.
Accuracy ranged from 0.8427 in clauses under 200 characters to 0.9254 in the
smallest-by-count longest-text bucket (≥2,000 characters, n=389).

Validation accuracy increased from 0.7634 for 3,445 examples with similarity
below 0.50 to 0.9816 for 816 examples at or above 0.98. Existing leakage audits
found many high-similarity and normalized exact overlaps between the supplied
splits. Similarity-stratified metrics are diagnostic context, not a selection
criterion or proof of causal memorization.

Generated JSON/Markdown reports contain complete per-class results, confusion
matrices, high-confidence errors, and length/similarity buckets under
`ml/reports/baseline/`; generated reports are gitignored.

## Experiment metadata and artifacts

`ExperimentMetadata` records model name/version, dataset and cached revision,
split, configuration, seed, and optional Git commit. `EvaluationResult`
stores metrics, per-class values, top-k scores, the full confusion matrix, and
top confused pairs. `resolve_git_commit` is best-effort and never blocks
execution.

The selected sklearn artifact contains the fitted vectorizer and classifier
together. Reports and model files are local generated outputs and are not
committed.

## Known limitations

- LEDGAR labels are imbalanced and single-label even when a clause contains
  several topics.
- Some categories have near-overlapping names and content; label boundaries
  need domain review.
- No source-contract IDs are exposed by the packaged dataset, so
  contract-level leakage cannot be ruled out.
- Results are specific to this SEC-contract-derived dataset and its supplied
  chronological split.
- LinearSVC scores are not calibrated probabilities.
