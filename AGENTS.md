# AGENTS.md

## Project Overview

This project studies binary logistic regression when the *training* data contains observations with missing labels.

- Response variable: $Y \in \{0, 1\}$ (true, unobserved when missing)
- Observed labels: $Y_{obs} \in \{0, 1, -1\}$ where $-1$ denotes a missing label
- Missingness indicator: $S \in \{0, 1\}$, where $S=1$ means the label is missing

The work is organized around three tasks: dataset preparation + missing-label simulation, a from-scratch Logistic Lasso solver via FISTA, and methods that leverage unlabeled observations.

## What Will Be Implemented

### Task 1 — Datasets + Missing-Label Schemes

- Collect 4 real binary classification datasets with mostly numerical features (multi-class may be converted to binary).
- Prepare datasets:
	- Impute missing values in $X$.
	- Remove collinear variables.
- Implement label-missingness generators that take fully observed $(X, Y)$ and return $(X, Y_{obs})$ under:
	- **MCAR** (controlled by missingness rate $c$)
	- **MAR1** (depends on a single feature)
	- **MAR2** (depends on all features)
	- **MNAR** (depends on $X$ and $Y$)

### Task 2 — Logistic Lasso via FISTA (Core Algorithm)

Implement Logistic Regression with an $\ell_1$ penalty (Logistic Lasso) using **FISTA**.

- Works on data with no missing values in $X$ or $y$.
- Inputs: training and validation splits.
- Select the best regularization strength $\lambda$ by optimizing a chosen validation metric.
- Supported validation measures:
	- recall, precision, F-measure (F1)
	- balanced accuracy (threshold 0.5)
	- ROC AUC
	- area under the sensitivity–precision curve
- Required API:
	- `fit()`, `predict_proba()`, `validate()`, `plot()` (metric vs. $\lambda$), `plot_coefficients()` (coefficients vs. $\lambda$)
- Include a comparison/discussion vs `sklearn.linear_model.LogisticRegression(penalty='l1', ...)`.

### Task 3 — Using Unlabeled Observations (UnlabeledLogReg)

Implement `UnlabeledLogReg` that uses both labeled and unlabeled observations from $(X, Y_{obs})$.

- Based on the Task 2 FISTA implementation.
- Propose two different algorithms for completing/inferring missing $Y$ prior to fitting.
- Compare:
	- two `UnlabeledLogReg` approaches
	- naive baseline (fit on labeled-only, $S=0$)
	- oracle (fit on true $(X, Y)$)
- Evaluate on a test set (missing labels only in training): accuracy, balanced accuracy, F1, ROC AUC.
- Analyze performance across the four missingness schemes and varying $c$ under MCAR.

## Repository Layout (Planned)

This repo currently contains only top-level documentation. Code will be added with a minimal, readable structure:

```
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── src/
│   ├── data/          # dataset loading, preprocessing, missing-label generation
│   ├── models/        # FISTA logistic lasso + UnlabeledLogReg
│   ├── metrics/       # metric implementations (AUC, balanced acc, etc.)
│   └── experiments/   # scripts to reproduce comparisons/plots
└── data/              # local cached datasets (optional; can be gitignored)
```

## Environment Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Code Style

- Prefer clean, readable code over inline comments
- Use docstrings for public functions/classes
- Type hints encouraged
- Follow PEP 8

## Notes

- Task 2 is the reusable core optimizer/model; Tasks 1 and 3 build experiments around it.


