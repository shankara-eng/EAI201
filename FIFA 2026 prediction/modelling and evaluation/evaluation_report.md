# Task 3 — Model Evaluation Report

Date: 2025-11-08

## Objective
Evaluate model performance on test data using accuracy, precision, recall, F1-score and ROC-AUC; produce confusion matrices and ROC curves; compare models and recommend a model choice with practical implications.

---

## Models evaluated
- Random Forest (100 trees)
- Logistic Regression (standardized features)

Visual artifacts generated (in workspace):
- `roc_curves_comparison.png`
- `confusion_matrices_comparison.png`
- `feature_importances_summary.csv` (for Task 4)

---

## Numeric results (test set)

Summary table (rounded):

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC  |
|---------------------|:--------:|:---------:|:------:|:-----:|:--------:|
| Random Forest       | 0.991    | 1.000     | 0.981  | 0.991 | 0.9997   |
| Logistic Regression | 0.995    | 0.991     | 1.000  | 0.995 | 0.9999   |

Notes:
- Metrics above computed on the held-out test set (20% split). ROC-AUC values are very near 1.0 because the dataset and labels are highly separable (see discussion below).

Confusion matrices (from `confusion_matrices_comparison.png`):

Random Forest

```
[[217   0]
 [  4 212]]
```

Logistic Regression

```
[[215   2]
 [  0 216]]
```

Interpretation of confusion matrices:
- Random Forest made 4 false negatives (actually Team B but predicted Team A?) and 0 false positives for Team A; Logistic Regression made 2 false positives and 0 false negatives. (Refer to printed matrices and code labeling for orientation.)

---

## ROC curves
See `roc_curves_comparison.png` — both models produce ROC curves that essentially reach the top-left corner, with AUCs ≈ 0.9997–0.9999. This indicates near-perfect separability under the current dataset/labels.

---

## Critical comparison — strengths, weaknesses, practical implications

Strengths
- Both models perform excellently on this dataset (high accuracy, F1 and AUC). Logistic Regression is slightly better on the held-out test set and is more interpretable.
- Random Forest can capture non-linear interactions and may be preferable if you later add features with non-linear relationships.

Weaknesses & Caveats
- Label leakage / synthetic-label effect: labels were computed from `A_points`, `B_points` and goals difference (score = A_points - B_points + 0.01*(goals_diff)). Because these exact or closely related features are fed into the model, the models can learn the deterministic rule and achieve near-perfect separation. Thus the very high ROC-AUC is expected and does not necessarily reflect ability to predict real match outcomes.
- Dataset composition: dataset contains all ordered team-pairs; the same team appears many times across samples. This can inflate apparent performance since team-level signatures occur in both train and test splits.
- Operational impact of errors:
  - False negatives (predicting A loses when A would win) might cause underestimation of stronger teams — in tactical terms, missing a strong contender could influence betting or team-selection strategies.
  - False positives (predicting A wins when it loses) could cause overconfidence.
  In this dataset the false negative/positive counts are tiny, but in realistic data these errors matter more.

Practical implications
- For production predictions on real matches, prefer simpler, interpretable models (Logistic Regression) if performance is similar, because interpretability helps explain and trust predictions.
- However, if you add richer features (player availability, injuries, recent head-to-head form), an ensemble model (Random Forest / Gradient Boosting) can capture complex patterns.

---

## Model choice recommendation
Given the current evaluation results and considering interpretability and comparable performance, **Logistic Regression** is recommended as the preferred model for deployment in this exercise because:
- It achieves the best overall test metrics (slightly higher accuracy and F1 here).
- Coefficients are interpretable and map to domain insights (see Task 4 feature interpretation).
- It is simpler and less likely to overfit if the dataset grows noisier or labels become real match outcomes.

Caveat: This recommendation assumes you will remove label-leakage or move to real match outcomes. If you plan to keep the synthetic label form but expand features with non-linear interactions, re-evaluate and consider Random Forest or more advanced ensembles.

---

## Next steps (recommended)
1. Replace synthetic labels with real match outcomes (actual match winners) or remove features used to construct labels (e.g., drop `A_points`/`B_points`) to obtain a realistic evaluation.
2. Use GroupKFold or team-based holdout to avoid team-level leakage between train and test.
3. Re-run evaluation and update the report with new metric tables and plots.


**Files generated**: `roc_curves_comparison.png`, `confusion_matrices_comparison.png`, `feature_importances_summary.csv`, `feature_importance_rf.png`, `feature_coefficients_lr.png`.


---

_End of report._
