# Evaluation

This document defines ClauseGuard's evaluation protocol for the LEDGAR
clause-categorization task. It covers split discipline, the metric set, the
shared evaluation harness, and the extension points planned for later phases.

## Split discipline

The LexGLUE LEDGAR package ships three predetermined splits
(60,000 / 10,000 / 10,000, chronological by filing year). We treat them as
strict, non-interchangeable roles:

```text
TRAIN      -> fitting only
VALIDATION -> model comparison, hyperparameter selection,
              threshold selection, error analysis
TEST       -> final held-out evaluation
```

Rules:

- Fit anything (vectorizers, imputation, class priors) only on `train`.
- Use `validation` for model comparison, hyperparameter selection, and any
  threshold tuning. This is the *development set*.
- The **test set must never be used for model selection** of any kind: no
  hyper-parameter choice, no threshold decision, no early stopping criterion
  based on test scores. It is evaluated exactly once per model, after the
  model and its configuration are frozen.
- Error analysis (confusion matrices, per-class failures) is performed on
  `validation`. Inspecting test errors for *reporting* is fine; adapting the
  model because of them is not.
- Similarity-to-training features (see the dataset audit) are dropped from the
  decision process: they are diagnostic context, not model inputs.

## Metrics

### Overview

The LEDGAR subset is single-label multiclass with 100 strongly imbalanced
classes (`Governing Laws` ≈ 4,243 examples; the rarest class has 25). The
metric set is chosen to represent the model's behaviour honestly under this
imbalance.

| Metric | Role |
| --- | --- |
| Macro-F1 | **Primary.** Averages per-class F1 equally, so frequent classes cannot dominate. |
| Accuracy | Overall share of correctly classified provisions. Intuitive but dominated by frequent classes. |
| Weighted-F1 | F1 averaged with each class weighted by its support; closer to "expected F1 over a random provision". |
| Per-class P / R / F1 / support | Diagnosis: shows which classes the model can and cannot separate. |
| Top-k accuracy (k=1,3,5) | Ratio of examples whose true class is within the top-k predictions. Useful for exploratory/assistive clause categorization. |
| Confusion matrix | Full 100×100 matrix plus top-N confused true/predicted pairs. |

### Why macro-F1 is the primary metric

Macro-F1 treats every class as equally important. With extreme class
imbalance, a model can reach high accuracy by memorising frequent classes and
ignoring rare ones; macro-F1 makes that failure visible and is the number we
care most about. All model-selection decisions will be made using
`validation` macro-F1 first.

### Micro-F1 is intentionally not reported as a primary metric

For single-label multiclass classification, micro-F1 and accuracy are equal by
construction, so reporting both as primary metrics would display redundant
information. Accuracy is kept because it is the more familiar formulation of
the same quantity.

### Top-k for exploratory categorization

Top-k answers a practical question for a *drafting assistant*: "is the correct
class among the top-k suggestions?" It uses the same score arrays (probabilities
or decision scores) and costs nothing extra. It is a secondary metric; the
primary decision metric remains macro-F1.

## Shared evaluation harness

The harness lives in `ml/src/evaluation/` and is small and model-agnostic:

- `metrics.py` — `classification_metrics`, `per_class_metrics`,
  `top_k_metrics`, `confusion_matrix_data`, `top_confused_pairs`.
  Top-k accepts either probability arrays or decision-score arrays, so
  `LogisticRegression` and `LinearSVC` can be evaluated with the same call.
- `metadata.py` — `ExperimentMetadata` (model name/version, dataset,
  revision, split, configuration, seed, optional `git_commit`) and
  `EvaluationResult` (structured, JSON-serializable via `to_json`).

Both the TF-IDF baseline and any transformer use this harness and the
`git_commit` from `resolve_git_commit` (best-effort; never fails CI).

## Extension points (future, not yet implemented)

These are intentionally deferred until the first baseline defines exactly how
model outputs are represented:

- **Calibration / ECE** — useful when confidence is presented to end users.
  Deferred: needs the chosen model's probability semantics.
- **Confidence-based abstention** — allowing the tool to decline low-confidence
  predictions (a drafting assistant benefits from "I'm not sure over a wrong
  guess"). Deferred: needs calibrated probabilities and a product decision on
  the abstention gate.
- **Similarity-stratified evaluation** — analysing model performance within
  cosine-similarity buckets to the training data (see the leakage audit
  artifacts: `<0.50`, `0.50–0.80`, `0.80–0.90`, `0.90–0.95`, `0.95–0.98`,
  `>=0.98`). This tells us *which* provisions the model actually generalises vs.
  memorises. Deferred: the bucket scripts and per-bucket metric reporting will be
  added alongside the baseline so the format matches real model outputs.

None of these become part of the harness architecture until the first baseline
exists.