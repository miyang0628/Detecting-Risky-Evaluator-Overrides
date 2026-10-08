# Simulated Override Selection Protocol

Replication code for:

> **Design-Induced Selection in Simulated Credit-Rating Overrides:
> A Cautionary Analysis and Validation Protocol**
> *[Journal withheld for blind review]*
> *[Authors withheld for blind review]*

---

## Overview

Credit rating agencies operating under regulatory frameworks are legally
required to let human evaluators review and potentially override
system-generated credit grades. Practitioners observe that evaluators tend to
*upgrade* borderline or distressed firms on qualitative grounds, raising the
question of whether such upgrade overrides carry incremental default risk.
Because proprietary evaluator override records are rarely accessible,
researchers often **simulate** the override step on public data.

This repository shows that the common simulation strategy is vulnerable to a
**design-induced selection effect** that can manufacture the very finding it
purports to test, and it provides a reproducible protocol for avoiding it. The
pipeline:

1. Trains a system-level credit grading model on financial variables and maps
   the estimated default probability to a seven-grade scale.
2. Simulates evaluator overrides under **three data-generating mechanisms
   (DGMs)** that differ *only* in how the upgrade probability is assigned —
   one tied to the system grade (and thus to the system-estimated default
   probability), and two tied to signals near-orthogonal to it.
3. Quantifies the upgrade–default association and shows how it collapses once
   the circularity is removed.
4. Validates robustness via repeated-simulation inference, parameter
   sensitivity, and leakage-aware feature-set comparison.
5. Uses an XGBoost + SHAP diagnostic to show that the override decision
   variables add negligible predictive value.

> **Note on framing.** This is a *cautionary* methodological study. Its
> conclusions concern what simulation designs can and cannot establish; it is
> **not** a claim about how real evaluators behave. All overrides are
> simulated.

---

## Key Findings

- When upgrades are tied to the system grade, upgraded firms default at a rate
  about **2.7 percentage points** above non-overridden firms. Once this
  circularity is removed (risk-neutral and pd-independent drivers), the gap
  **falls within sampling error of zero** across 500 replications.
- The grade-adjustment odds ratio, controlling for the system-estimated
  default probability, **includes unity** under all three mechanisms.
- Holding the model class fixed, override features add only **+0.0008 ROC-AUC**
  over a financial baseline; **redeveloping** a separate model per feature set
  from scratch gives the same conclusion (ΔAUC ≈ 0).
- A large apparent gain reported in naive designs (e.g. +0.206 AUC) is
  attributable to switching model class (logistic → gradient boosting), not to
  the override information.

---

## Dataset

**Polish Companies Bankruptcy Dataset**
- Source: UCI Machine Learning Repository (ID: 365)
- 43,405 firm-year observations across 5 forecast horizons
- 64 financial features + binary bankruptcy label
- Overall default rate: 4.82%

**Download:**
```
https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
```
Place `1year.arff` – `5year.arff` in `data/raw/` before running. The
pre-processed `.parquet` files under `data/processed/` are **regenerated** by
running `NB01`–`NB04` and are excluded from version control (see
`.gitignore`) to keep the repository lightweight.

---

## Repository Structure

```
simulated-override-selection-protocol/
├── data/
│   ├── raw/                         ← place .arff files here
│   └── processed/                   ← pre-generated parquet files
├── notebooks/
│   ├── NB01_data_preparation.ipynb
│   ├── NB02_eda.ipynb
│   ├── NB03_grade_assignment.ipynb
│   ├── NB04_diag_correlation.ipynb
│   ├── NB04_override_simulation.ipynb
│   ├── NB04b_override_simulation_independent.ipynb
│   ├── NB04c_override_simulation_replications.ipynb
│   ├── NB04d_override_simulation_sensitivity.ipynb
│   ├── NB05_statistical_model.ipynb
│   ├── NB06_ml_model.ipynb
│   ├── NB06b_validation_protocol.ipynb
│   ├── NB07_shap_explainability.ipynb
│   ├── NB08_evaluation.ipynb
│   ├── NB09_sensitivity_analysis.ipynb
│   ├── NB10_robustness_summary.ipynb
│   ├── NB11_evaluator_pattern_analysis.ipynb
│   ├── NB12_collinearity_diagnostics.ipynb      ← NEW (revision)
│   ├── NB13_grade_construction.ipynb            ← NEW (revision)
│   ├── NB14_attr43_profile.ipynb                ← NEW (revision)
│   ├── NB15_xgb_tuning.ipynb                    ← NEW (revision)
│   └── NB16_shap_three_models.ipynb             ← NEW (revision)
├── results/
│   ├── figures/                     ← 600-dpi PNG + vector PDF
│   └── tables/                      ← CSV
├── models/                          ← fitted logistic / XGBoost models
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Revision Notebooks (reviewer response)

Five notebooks were added to address the peer-review comments. Each is written
as a single self-contained cell, saves every figure as both 600-dpi PNG and
vector PDF in grayscale, and writes its tables to `results/tables/`.

| Notebook | Addresses | Main outputs |
|----------|-----------|--------------|
| `NB12_collinearity_diagnostics` | Multicollinearity of the 64-variable system model | VIF table, correlation clustermap (fig. 29–30, tbl. 19) |
| `NB13_grade_construction` | Justification of seven grades and boundary methodology | grade profile, 5/7/10/14-grade comparison (fig. 31, tbl. 20) |
| `NB14_attr43_profile` | Description of `Attr43` and its cross-matrix with the system grade | bad-rate heatmap, orthogonality stats (fig. 32, tbl. 21) |
| `NB15_xgb_tuning` | Hyperparameter tuning rationale vs. fixed settings | tuned-vs-fixed comparison (fig. 36, tbl. 22) |
| `NB16_shap_three_models` | Full redevelopment + SHAP for all three feature sets | per-model AUC and SHAP (fig. 33–35, tbl. 23) |

---

## Reproducing the Results

```bash
# 1. Create an environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) regenerate processed data from the raw .arff files
#    Run notebooks in order NB01 → NB16.

# 3. Run any notebook non-interactively, e.g.:
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/NB12_collinearity_diagnostics.ipynb
```

All notebooks use relative paths (`../data`, `../results`) and are intended to
be run from the `notebooks/` directory. Random seeds are fixed (`random_state =
42`) throughout for reproducibility.

---

## Requirements

See `requirements.txt`. Core stack: `numpy`, `pandas`, `scikit-learn`,
`xgboost`, `shap`, `statsmodels`, `matplotlib`, `seaborn`.

---

## License

See `LICENSE`.

---

*This repository is anonymized for blind peer review. Author and institution
details have been removed and will be restored upon acceptance.*
