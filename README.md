# AML_project1

Binary logistic regression with missing labels (Project 1).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Task 2 (Logistic Lasso via FISTA)

Core implementation lives in `src/models/logistic_lasso_fista.py`.

### Run smoke tests

```bash
./.venv/bin/python tests/smoke/test_metrics_smoke.py
./.venv/bin/python tests/smoke/test_fista_smoke.py
```

### Compare vs scikit-learn

```bash
./.venv/bin/python -m src.experiments.task2_sklearn_comparison --metric roc_auc
```

Optional: save plots to a folder

```bash
./.venv/bin/python -m src.experiments.task2_sklearn_comparison --metric balanced_accuracy --save-dir outputs
```
