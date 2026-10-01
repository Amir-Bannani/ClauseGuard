# ClauseGuard — Classification Model Improvement & Error Analysis

## 1. Model Improvement

After cleaning the dataset and adding synthetic examples, the clause classification model improved significantly, especially on **Macro F1**, which is important because the dataset is imbalanced across the 10 classes.

| Metric              | Before |      After |       Change |
| ------------------- | -----: | ---------: | -----------: |
| Validation Accuracy | 88.76% |     90.23% |     +1.47 pp |
| Validation Macro F1 | 77.40% | **85.89%** | **+8.49 pp** |
| Test Accuracy       | 88.31% |     87.76% |     -0.55 pp |
| Test Macro F1       | 76.93% | **80.46%** | **+3.53 pp** |
| Test Weighted F1    | 87.64% |     87.80% |     +0.16 pp |

### Main takeaway

The most important result is the **+3.53 percentage-point improvement in Test Macro F1**.

This indicates that the model became more balanced across the different legal clause categories, rather than simply improving performance on the majority classes.

The small decrease in test accuracy (~0.55 pp) is not necessarily concerning, especially given the relatively small test set (539 examples). Macro F1 provides a more informative view of performance for this imbalanced classification task.

---

## 2. `no_solicit_customers` Improvement

The `no_solicit_customers` class was previously completely missed:

**Before:**

* F1 = **0%**

**After:**

* F1 = **36.36%**

The model is now able to identify some customer non-solicitation clauses, suggesting that the cleaned data and synthetic examples improved its ability to recognize this category.

However, the class remains relatively weak and requires further investigation.

---

## 3. Confusion Matrix Analysis

The test confusion matrix showed that the remaining errors are **structured rather than random**.

For `no_solicit_customers`:

* 10 true examples in the test set
* **4 correctly classified**
* 4 classified as `non_compete`
* 2 classified as `exclusivity`

Therefore, the main issue is distinguishing **restrictive covenant categories** that use overlapping legal language.

### Main confusion relationships

```text
                 ┌──────────────────┐
                 │ no_solicit       │
                 │ customers        │
                 └────────┬─────────┘
                          ↕
                   ┌──────┴──────┐
                   │ non_compete │
                   └──────┬──────┘
                          ↕
                   ┌──────┴──────┐
                   │ exclusivity │
                   └─────────────┘
```

Observed errors included:

* `no_solicit_customers` → `non_compete`: 4
* `no_solicit_customers` → `exclusivity`: 2
* `non_compete` → `exclusivity`: 7
* `non_compete` → `no_solicit_customers`: 2

This indicates substantial semantic overlap between these categories in the training/test data.

---

## 4. Interpretation of the Three Classes

The model should learn to distinguish the **specific restriction being imposed**, rather than relying only on general restrictive language.

### `no_solicit_customers`

Core concept:

> Restriction on approaching, soliciting, diverting, or attempting to take customers/clients.

Typical signals:

* solicit customers
* solicit clients
* approach clients
* divert customers
* induce customers to leave

### `non_compete`

Core concept:

> Restriction on working for, operating, or participating in a competing business.

Typical signals:

* compete with the employer
* work for a competitor
* operate a competing business
* engage in competing activities

### `exclusivity`

Core concept:

> Requirement to provide services/work exclusively to one party or restriction on outside professional activities.

Typical signals:

* exclusively provide services
* devote full time to the employer
* no outside employment
* cannot engage in other professional activities

---

## 5. Current Decision

**Do not increase the number of epochs yet.**

The overall model has reached a healthy baseline, and the confusion matrix provides a more useful direction for the next improvement.

The next optimization should focus on **class-specific data quality**, particularly the distinction between:

```text
no_solicit_customers
        ↕
non_compete
        ↕
exclusivity
```

Potential next steps:

1. Inspect the actual misclassified examples.
2. Identify recurring linguistic patterns causing the confusion.
3. Check whether some CUAD examples have ambiguous or overlapping labels.
4. Improve synthetic examples specifically for these distinctions.
5. Retrain and compare Macro F1 and the confusion matrix.
6. Avoid increasing epochs unless there is evidence of underfitting.

## Overall Status

The model is **substantially better than the previous version**, particularly in balanced class performance.

The remaining weakness appears to be primarily a **class-definition / data-separation problem**, not simply a training-duration problem.
