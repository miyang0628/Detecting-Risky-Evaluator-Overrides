# Detecting Risky Evaluator Overrides in Corporate Credit Rating

Replication code for:

> **[Title withheld for blind review]**
> *[Journal withheld for blind review]*

---

## Overview

Credit rating agencies are legally required to have human evaluators review
and potentially override system-generated credit grades. While this two-stage
process is mandated by regulation, it introduces a systematic risk: evaluators
tend to upgrade borderline or distressed firms based on qualitative judgment,
and those upgraded firms subsequently default at higher rates than the assigned
grade implies.

This repository provides a fully reproducible pipeline that:

1. Trains a system-level credit grading model on financial variables
2. Simulates the evaluator override process under regulatory constraints
3. Quantifies the incremental default risk of upgrade overrides
4. Builds an explainable early-warning ML model to flag risky overrides
5. Identifies financial signals overlooked in risky upgrade decisions
6. Validates robustness across multiple simulation scenarios

---

## Key Findings

- Upgrade overrides are associated with a **58% higher default rate**
  (7.69% vs 4.86% for non-overridden cases)
- A one-notch upgrade increases the odds of default by **7.98%**
  (OR = 0.926, p = 0.005) after controlling for financial variables
- Override-augmented XGBoost achieves **ROC-AUC = 0.995**
  (ΔAUC = +0.206 vs financial-only baseline)
- **Risky upgrade cases** (upgraded → defaulted) exhibit disproportionately
  high SHAP contributions from three financial distress signals:
  interest coverage ratio (Attr27), quick ratio (Attr46),
  and cost efficiency ratio (Attr58)
- **Grade transitions originating from CCC** carry the highest residual
  default risk: CCC → BB (19.6%) and CCC → B (17.4%), both far above
  the upgrade-group average of 7.69%
- Findings are **robust** across conservative, base, and aggressive
  simulation scenarios

---

## Dataset

**Polish Companies Bankruptcy Dataset**
- Source: UCI Machine Learning Repository (ID: 365)
- Reference: Zieba, M., Tomczak, S.K., & Tomczak, J.M. (2016).
  *Ensemble Boosted Trees with Synthetic Features Generation in
  Application to Bankruptcy Prediction.* Expert Systems with Applications,
  58, 93–101.
- 43,405 firm-year observations across 5 forecast horizons
- 64 financial features + binary bankruptcy label
- Overall default rate: 4.82%

**Download:**
```
https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
```
Place `1year.arff` – `5year.arff` in `data/raw/` before running.

---

## Repository Structure

```
credit_override_study/
├── data/
│   ├── raw/                        ← place .arff files here
│   └── processed/                  ← auto-generated parquet files
├── notebooks/
│   ├── NB01_data_preparation.ipynb
│   ├── NB02_eda.ipynb
│   ├── NB03_grade_assignment.ipynb
│   ├── NB04_override_simulation.ipynb
│   ├── NB05_statistical_model.ipynb
│   ├── NB06_ml_model.ipynb
│   ├── NB07_shap_explainability.ipynb
│   ├── NB08_evaluation.ipynb
│   ├── NB09_sensitivity_analysis.ipynb
│   ├── NB10_robustness_summary.ipynb
│   └── NB11_evaluator_pattern_analysis.ipynb
├── models/
│   ├── logistic/
│   └── xgboost/
├── results/
│   ├── figures/
│   └── tables/
├── utils/
│   └── helpers.py
├── requirements.txt
└── README.md
```

---

## Notebook Pipeline

| Notebook | Purpose |
|----------|---------|
| NB01 | Load `.arff` files, clean, impute, save master parquet |
| NB02 | Exploratory data analysis: class balance, distributions, correlations |
| NB03 | Logistic regression → system Pd → quantile-based grade assignment |
| NB04 | Evaluator override simulation (grade-boundary optimistic bias) |
| NB05 | WoE / Information Value + statsmodels logit with p-values |
| NB06 | XGBoost (5-fold CV) with override-augmented features |
| NB07 | SHAP: global summary, dependence plot, individual force plots |
| NB08 | Final evaluation: ROC/PR curves, calibration, business impact |
| NB09 | Sensitivity analysis across three simulation scenarios |
| NB10 | Robustness summary and cross-scenario comparison |
| NB11 | Evaluator pattern analysis: financial signals in risky upgrade cases |

**Run order:** NB01 → NB02 → NB03 → NB04 → NB05 → NB06 → NB07 → NB08 → NB09 → NB10 → NB11

> **Note:** NB11 depends on outputs from NB04 (`override_data.parquet`)
> and NB06 (`xgb_override.pkl`). Run those notebooks first.

---

## NB11: Evaluator Pattern Analysis

NB11 addresses the question: *"What financial signals are systematically
present in risky upgrade cases but appear to be overlooked by evaluators?"*

This notebook conducts three analyses:

| Analysis | Question | Output |
|----------|----------|--------|
| A | Which financial signals concentrate in risky upgrades? | Fig 26 |
| B | Which grade transitions carry the highest default risk? | Fig 27 |
| C | How do financial profiles differ between risky and safe upgrades? | Fig 28 |

**Key outputs:**
- `results/figures/26_shap_risky_upgrade_top10.png`
- `results/figures/27_grade_transition_default_rate.png`
- `results/figures/28_financial_signal_comparison.png`
- `results/tables/NB11_pattern_summary.csv`
- `results/tables/NB11_transition_summary.csv`

> **Important:** NB11 operates at the *firm level*, not the *evaluator level*.
> It identifies portfolio-level outcome patterns associated with upgrade
> overrides. It does not identify individual evaluator behaviour or attribute
> override decisions to specific personnel.

---

## Environment Setup (Windows / Anaconda)

```bash
# 1. Create environment
conda create -n credit_override_env python=3.10 -y

# 2. Activate
conda activate credit_override_env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register Jupyter kernel
python -m ipykernel install --user \
    --name credit_override_env \
    --display-name "credit_override"

# 5. Launch Jupyter
cd path\to\credit_override_study
jupyter notebook
```

**Important:** After opening a notebook, select kernel →
**credit_override** before running.

---

## Dependencies

```
numpy>=1.24        pandas>=2.0        scipy>=1.10
scikit-learn>=1.3  xgboost==1.7.6     shap>=0.44
statsmodels>=0.14  matplotlib>=3.7    seaborn>=0.12
joblib>=1.3        pyarrow>=14.0      ipykernel>=6.0
```

> **Note on XGBoost version:** `xgboost==1.7.6` is pinned for SHAP
> compatibility. XGBoost ≥ 2.0 stores `base_score` as `[5E-1]`, which
> SHAP cannot parse without a source-level patch. NB07 and NB11 both
> apply this patch automatically if a newer XGBoost version is detected.

---

## Reproducibility

All random seeds are fixed (`numpy.random.seed(2024)`, `random_state=42`).
Running notebooks in order (NB01 → NB11) produces identical results.

---

## Note on pd_system and Data Leakage

`pd_system` (the system model's predicted probability of default) is included
as a feature in NB05–NB08 and NB11. This is intentional and does **not**
constitute data leakage: in real-world credit rating workflows, evaluators
have full access to the system-generated Pd score **before** making their
override decision. Including `pd_system` therefore reflects the information
set available at the time of the override, not future information.

---

## Note on Analysis Level

All analyses in this repository are conducted at the **firm level**
(unit of observation: firm-year record). The findings describe
portfolio-level outcome patterns — which types of firms, override
directions, and grade transitions are associated with elevated default
rates. The repository does **not** contain evaluator-level identifiers,
qualitative override reasoning, or institutional context. Evaluator
behaviour cannot be attributed or corrected from these analyses alone.

---

## Citation

If you use this code, please cite:

```bibtex
@article{anonymous2025override,
  title   = {[Title withheld for blind review]},
  author  = {Anonymous},
  journal = {[Journal withheld for blind review]},
  year    = {2025},
}
```

---

## License

This code is released for academic replication purposes.
The dataset is subject to the UCI ML Repository terms of use.
