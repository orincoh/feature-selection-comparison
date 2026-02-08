# Feature Selection Methods — Empirical Comparison (Filter / Wrapper / Hybrid)

This project empirically compares **filter**, **wrapper**, and **hybrid** feature selection approaches on multiple classification datasets.
All methods are evaluated with a **Random Forest** classifier and tuned using **RandomizedSearchCV** to maximize **F1-macro**.

Author: Orin Cohen  
Course: Selected Topics in Learning Systems (M.Sc.)

---

## Methods Compared

### Filter methods
- Mutual Information (MI)
- Symmetrical Uncertainty (SU)
- ReliefF
- ANOVA F-test (continuous datasets only)
- Chi-Square (categorical datasets only)

Filter methods select features using a **log2(n)** heuristic (n = number of features).

### Wrapper methods
- RFECV (Recursive Feature Elimination with CV)
- SFS (Sequential Forward Selection)
- Backward Elimination

Wrappers optimize feature subsets based on Random Forest performance using **F1-macro**.

### Hybrid methods
- Hybrid 1: Filter (MI) ➜ Wrapper (RFECV)
- Hybrid 2: Weighted combination of MI + RF feature importance (embedded)

---

## Datasets

Six benchmark datasets were used with diverse characteristics (numeric / categorical / mixed and binary / multiclass):
- Breast Cancer Wisconsin (Diagnostic) (scikit-learn)
- Iris (scikit-learn)
- Wine (scikit-learn)
- Mobile Price Classification (CSV)
- Mushrooms (CSV, one-hot encoded)
- Zoo (CSV, one-hot encoded)

Train/test split: **80% / 20% stratified**.  
For Mushrooms and Zoo, one-hot encoding is applied **after the split** to prevent leakage.

---

## Experimental Setup

- Model: RandomForestClassifier (class_weight='balanced')
- Tuning: RandomizedSearchCV, scoring = **f1_macro**
- Metrics reported on test set: **F1-macro, Recall-macro, Precision-macro**
- Fixed random seed: **42** for reproducibility

---

## Results (high-level)

Overall, there is **no single best feature selection approach** across all datasets.
Wrappers and hybrids can achieve higher performance but may require substantially longer runtime (especially on larger datasets).

For detailed tables (per method, per dataset) see the included PDF report.

---

## Repository Structure

- `src/main.py` — full pipeline: data prep ➜ feature selection ➜ RF training/tuning ➜ metrics ➜ plots
- `data/` — CSV datasets used in the project
- `results/feature_selection_results.json` — selected feature sets per dataset and method (optional to commit)
- `plots/` — generated feature-importance plots (created automatically)

---

## How to Run

### 1) Create environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
