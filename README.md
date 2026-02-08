# Feature Selection Methods – Comparative Study

A small empirical study comparing filter, wrapper and hybrid feature selection methods on several classification datasets.
All methods are evaluated with the same setup: Random Forest + RandomizedSearchCV (F1-macro).

By: Orin Cohen  
Course: Selected Topics in Learning Systems (M.Sc.)

---

## Summary

What’s in the project:
- Compare multiple FS strategies (filter / wrapper / hybrid) under consistent conditions.
- Report F1/Recall/Precision (macro) and runtime.
- Generate feature-importance plots per dataset and method.

What was found (see the PDF for full tables):
- There isn’t one method that wins on every dataset.
- Wrapper/hybrid methods can be a bit better on some datasets, but they’re much slower.
- On the Mobile dataset, the hybrid approach performed worse than filter/wrapper.
- On simple low-dimensional data (Iris), most approaches look very similar.

---

## Repository contents

- `src/main.py`  
  Full pipeline: load data → feature selection → model tuning → evaluation → plots.

- `data/`  
  CSV datasets included here:
  - `Mobile_Price.csv`
  - `mushrooms.csv`
  - `zoo.csv`  
  Breast Cancer, Iris and Wine are loaded via `scikit-learn` (see the report).

- `results/` (optional)  
  Saved artifacts from runs.  
  The code currently reads/writes `feature_selection_results.json`.

- `plots/`  
  Generated plots (created when running the code).

- `Feature Selection Methods Project Orin Cohen.pdf`  
  Full report (methods, tables, discussion).

---

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py

Note: the CSV files (Mobile_Price.csv, mushrooms.csv, zoo.csv) should be located in the repository root.
