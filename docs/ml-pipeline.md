# LEDGAR TF-IDF baseline

## Experiment status

The first classifier experiment is complete. It compares word TF-IDF with
multiclass Logistic Regression and LinearSVC on all 100 LexGLUE LEDGAR labels.
The selected configuration is LinearSVC with `C=3` and no class weighting,
chosen using validation macro-F1. It was evaluated once on the held-out test
split after selection. No transformer was trained.

## Data and split protocol

- Dataset: `coastalcph/lex_glue`, configuration `ledgar`.
- Cached dataset revision: `0.0.0`; the supplied chronological splits contain
  60,000 train, 10,000 validation, and 10,000 test examples.
- All 100 labels are preserved. No labels were merged or removed, and no
  resampling or synthetic examples were used.
- The largest class is Governing Laws (4,243 examples); the smallest is Books
  (25). The validation split has no Books examples. Macro-F1 therefore uses
  the fixed 100-label set, assigning zero F1 to a class with no validation
  support and no predictions.
- Tuning and error-analysis commands load only train and validation. The final
  command loads train and test after the selection is frozen. The vectorizer
  is fit on train only in every run.

The full class-support audit is in the ignored generated artifacts under
`ml/reports/`. The dataset package exposes no source-contract identifier, so
contract-level split separation cannot be verified. Existing audits also found
normalized exact overlaps and high TF-IDF similarity across the supplied
splits; similarity is reported as diagnostic context, not used for selection.

## Pipeline

`ml/src/data_loader.py` reads the cached Arrow files when available and loads
only requested splits. `ml/src/baseline.py` defines the vectorizer, classifiers,
training grids, prediction scoring, evaluation, metadata, and CLI. The fitted
vectorizer and classifier are saved together as one sklearn `Pipeline`.
`ml/src/error_analysis.py` produces validation-only diagnostic artifacts.
Metrics and result structures come from `ml/src/evaluation/`.

### TF-IDF

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32,
    lowercase=True,
)
```

No stop words are removed, and no stemming, lemmatization, or custom text
normalization is applied. Lowercasing is performed by the vectorizer; legal
tokens such as “shall”, “may”, “not”, “unless”, and “provided” remain available.
The audit reports a 173,477-term vocabulary under these settings.

### Logistic Regression

Multinomial `LogisticRegression` with L2 penalty, `lbfgs`, `max_iter=2000`,
`tol=1e-3`, and seed `20260922`. The tolerance was relaxed from sklearn's
default after full-data pilot fits showed that stricter optimization took
substantially longer; the final grid records convergence and iteration counts.
The controlled validation grid was:

| C | class_weight |
| --- | --- |
| 1, 3, 10, 30, 100 | `None`, `balanced` |

### LinearSVC

L2 penalty, squared-hinge loss, `dual="auto"`, `max_iter=5000`, `tol=1e-4`,
seed `20260922`, and `class_weight=None`. The validation grid was
`C ∈ {0.1, 0.3, 1, 3}`. It uses the same TF-IDF parameters and training split.
Top-k rankings use `decision_function` scores; these scores are not calibrated
probabilities.

## Validation results

Macro-F1 is computed over all 100 class IDs. Accuracy and weighted-F1 are
supporting metrics. Every fit below converged. Top-k values for LinearSVC are
based on decision-score ranking.

### Logistic Regression

| C | class weight | Accuracy | Macro-F1 | Weighted-F1 | Top-3 | Top-5 | Iterations |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | none | 0.8385 | 0.7448 | 0.8288 | 0.9489 | 0.9689 | 36 |
| 1 | balanced | 0.8364 | 0.7766 | 0.8397 | 0.9524 | 0.9743 | 30 |
| 3 | none | 0.8628 | 0.7869 | 0.8570 | 0.9615 | 0.9786 | 50 |
| 3 | balanced | 0.8517 | 0.7931 | 0.8541 | 0.9605 | 0.9784 | 40 |
| 10 | none | 0.8761 | **0.8068** | 0.8726 | 0.9656 | 0.9804 | 71 |
| 10 | balanced | 0.8522 | 0.7948 | 0.8548 | 0.9609 | 0.9772 | 37 |
| 30 | none | 0.8636 | 0.7907 | 0.8597 | 0.9646 | 0.9805 | 57 |
| 30 | balanced | 0.8630 | 0.8033 | 0.8631 | 0.9628 | 0.9805 | 47 |
| 100 | none | 0.8628 | 0.7928 | 0.8601 | 0.9646 | 0.9792 | 56 |
| 100 | balanced | 0.8539 | 0.7918 | 0.8553 | 0.9617 | 0.9778 | 46 |

Balanced weighting improved macro-F1 at C=1 and C=30, but reduced it at C=3,
10, and 100. It was not a consistent improvement.

### LinearSVC

| C | Accuracy | Macro-F1 | Weighted-F1 | Top-3 | Top-5 | Iterations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.8575 | 0.7712 | 0.8469 | 0.9612 | 0.9747 | 13 |
| 0.3 | 0.8750 | 0.7994 | 0.8679 | 0.9607 | 0.9742 | 21 |
| 1 | 0.8835 | 0.8185 | 0.8788 | 0.9580 | 0.9707 | 48 |
| 3 | 0.8828 | **0.8194** | 0.8795 | 0.9535 | 0.9660 | 129 |

The full per-class scores, confusion matrices, top confused pairs, fit times,
and configuration metadata are written to ignored JSON/Markdown artifacts in
`ml/reports/baseline/` when the commands below are run.

## Selection and final held-out results

The configuration selected under this protocol is LinearSVC, `C=3`,
`class_weight=None`. Its validation macro-F1 of 0.8194 was the highest among the
tested configurations. It exceeded the strongest Logistic Regression result
(C=10, no class weighting; 0.8068) by 0.0126. LinearSVC at C=1 had slightly
higher top-3 and top-5 accuracy, but macro-F1 was the predeclared selection
criterion.

After selection, the pipeline was fit on train and evaluated once on test:

| Metric | Test result |
| --- | ---: |
| Accuracy / Top-1 | 0.8797 |
| Macro-F1 (100 labels) | 0.8299 |
| Weighted-F1 | 0.8757 |
| Top-3 accuracy | 0.9506 |
| Top-5 accuracy | 0.9621 |

The held-out result is reported for the frozen configuration and was not used
to alter model selection. The per-class test metrics, full 100×100 confusion
matrix, and top confused pairs are in the generated `test_evaluation.json`.
The single fitted preprocessing-plus-classifier artifact is
`ml/artifacts/baseline_linearsvc_c3_0_none.joblib`; model artifacts are
gitignored.

On test, the weakest F1 values included Applicable Laws (0.30; support 53),
Assigns (0.40; support 4), Jurisdictions (0.43; support 29), Venues (0.44;
support 20), and Miscellaneous (0.53; support 79). The largest test confusions
were Applicable Laws → Governing Laws (31), No Waivers → Waivers (21), Tax
Withholdings → Withholdings (19), and Definitions → Defined Terms (18). These
patterns mirror the validation label-boundary confusions. LinearSVC C=3's
validation macro-F1 advantage over C=1 was only 0.0009, and C=1 had higher
top-k accuracy; the selected configuration follows the specified macro-F1
criterion. Test macro-F1 (0.8299) was above validation macro-F1 (0.8194), while
test accuracy was slightly lower (0.8797 versus 0.8828); this is reported as
split variation and did not trigger a configuration change.

## Validation error analysis

Diagnostics below use the selected Logistic Regression validation model
(C=10, unweighted), so confidence can be read as a probability. They were
completed before the LinearSVC comparison and did not use test examples.

- **Confused pairs:** Applicable Laws → Governing Laws was the most frequent
  directed error (50). Other recurring pairs included Defined Terms ↔
  Definitions (17 and 13), Warranties → Representations (17), Tax Withholdings
  ↔ Withholdings (16 and 14), Authorizations → Authority (15), and No Waivers
  ↔ Waivers (14 and 14).
- **Support and class performance:** Assigns had 31 training examples and 3
  validation examples, with zero recall. Qualifications had 47 training and 8
  validation examples (F1 0.36). Applicable Laws had 69 validation examples
  and recall 0.19. Conversely, high-support Counterparts (2,427 train / 429
  validation) reached F1 0.99, and Governing Laws (3,167 / 494) reached 0.91.
  Books had no validation examples; its score is zero in the fixed 100-label
  macro average but does not provide a validation estimate for that class.
- **Confident errors:** Some high-probability errors appear label-boundary
  related. For example, validation clauses explicitly about signing in
  counterparts were labeled Miscellaneous or Effectiveness and predicted as
  Counterparts. A short No Waivers example was predicted as Waivers. These
  examples are evidence of overlapping labels in the observed data, not proof
  that any individual label is wrong.
- **Clause length:** Accuracy ranged from 0.8427 for texts under 200 characters
  to 0.9254 for texts at least 2,000 characters. The latter bucket has only
  389 examples. Intermediate buckets ranged from 0.8733 to 0.8849.
- **Similarity to train:** Accuracy was 0.7634 for 3,445 validation examples
  below 0.50 similarity, and 0.9816 for 816 examples at or above 0.98. This is
  a strong association with similarity; the audit shows that near-duplicate
  boilerplate is common, so it should not be interpreted as causal evidence of
  generalization.
- **Manual review categories:** The inspected examples show semantic label
  overlap (Applicable Laws/Governing Laws; Defined Terms/Definitions),
  boilerplate with a clear topic word (counterparts), short text, and clauses
  combining more than one topic. Possible label noise and genuine model errors
  remain hypotheses for individual examples; this experiment does not adjudicate
  them.

Detailed validation examples and all diagnostic tables are in
`ml/reports/baseline/error_analysis/` (ignored by Git).

## Reproducibility

From the repository root, with `requirements.txt` installed:

```bash
python -m ml.src.baseline lr-grid
python -m ml.src.baseline error-analysis --model logreg --C 10 --class-weight none
python -m ml.src.baseline svc-grid
python -m ml.src.baseline final --model linearsvc --C 3 --class-weight none
```

The final command refuses to overwrite an existing test report. Experiment
metadata records the seed, dataset/configuration/revision, model version, full
parameters, split, and best-effort Git commit. Generated reports, cached data,
and the fitted model are intentionally not committed.

## Limitations and next experiment

The dataset's single labels force one category even where clauses discuss
several topics. Some class names and examples have close semantic boundaries.
Validation and test contain highly similar provisions, while the package lacks
contract IDs for source-level split checks. These results therefore describe
the supplied LexGLUE split and should not be generalized to arbitrary contract
collections. LinearSVC decision scores are ranking scores, not calibrated
confidence estimates.

The next experiment should first review the most-confused label pairs with
domain guidance, then test one controlled feature change (for example, adding
character n-grams) against this frozen word-TF-IDF baseline. Keep the same
train/validation/test protocol and do not tune the completed test result.
