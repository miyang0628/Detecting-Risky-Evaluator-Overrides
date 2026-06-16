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
5. Validates robustness across multiple simulation scenarios

---

## Key Findings

- Upgrade overrides are associated with a **58% higher default rate**
  (7.69% vs 4.86% for non-overridden cases)
- A one-notch upgrade increases the odds of default by **7.98%**
  (OR = 0.926, p = 0.005) after controlling for financial variables
- Override-augmented XGBoost achieves **ROC-AUC = 0.995**
  (ΔAUC = +0.206 vs financial-only baseline)
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
│   └── NB10_robustness_summary.ipynb
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

> **Note:** `xgboost==1.7.6` is required for SHAP compatibility.
> XGBoost ≥ 2.0 produces a `base_score` format (`[5E-1]`) that
> current SHAP versions cannot parse without a source-level patch
> (applied automatically in NB07).

---

## Reproducibility

All random seeds are fixed (`numpy.random.seed(2024)`, `random_state=42`).
Running notebooks in order (NB01 → NB10) produces identical results.

---

## Note on pd_system and Data Leakage

`pd_system` (the system model's predicted probability of default) is included
as a feature in NB05–NB08. This is intentional and does **not** constitute
data leakage: in real-world credit rating workflows, evaluators have full
access to the system-generated Pd score **before** making their override
decision. Including `pd_system` therefore reflects the information set
available at the time of the override, not future information.

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
