# Design-Induced Selection in Simulated Evaluator Overrides: A Cautionary Analysis and Validation Protocol

Replication code for:

> **[Title withheld for blind review]**
> *[Journal withheld for blind review]*

---

## Overview

Credit rating agencies operating under regulatory frameworks are legally
required to have human evaluators review and potentially override
system-generated credit grades. Practitioners observe that evaluators tend to
upgrade borderline or distressed firms on qualitative grounds. A natural
question is whether such upgrade overrides carry incremental default risk.

Because proprietary evaluator override records are rarely accessible, a common
strategy is to **simulate** the override step on a public dataset. This
repository shows that such simulations are prone to a **design-induced
selection effect**: when upgrade probability is tied to the system grade — which
is itself a transform of the system-estimated default probability
(`pd_system`) — the simulated upgrade group mechanically over-samples
high-risk firms, and the "upgrade risk" finding is largely an artifact of the
simulation rules rather than evidence of evaluator behaviour.

Rather than claiming an empirical finding, this repository provides a fully
reproducible pipeline that:

1. Trains a system-level credit grading model on financial variables
2. Simulates the override step under **three alternative data-generating
   mechanisms (DGMs)** that differ only in how upgrade probability is assigned
3. Quantifies how much of the apparent upgrade-risk effect is induced by the
   simulation design, using **repeated simulations** and **parameter
   sensitivity analysis**
4. Isolates the **pure incremental contribution** of override features under a
   leakage-aware, single-model comparison
5. Provides a **validation protocol** and documents the limitations that make
   real evaluator-level data indispensable for any causal claim

---

## Key Findings

- **The upgrade-risk effect is design-induced.** Under the naive DGM (upgrade
  probability driven by system grade), upgraded firms show a default-rate gap
  of **+3.2 percentage points** over non-overridden firms. When upgrade
  probability is instead driven by a signal near-orthogonal to `pd_system`,
  the gap collapses to **+0.6 pp (risk-neutral driver)** and **+0.1 pp
  (size driver)** — roughly **80% of the naive effect disappears**.
- **The residual effect is not statistically distinguishable from zero.**
  Across **500 replications**, the upgrade-vs-none default-rate gap has a 95%
  interval that **excludes zero only under the naive DGM**; both
  pd-independent DGMs include zero. The `grade_diff` odds ratio (controlling
  for `pd_system`) includes 1.0 under **all** DGMs.
- **The conclusion is robust to parameter choice.** Across a grid of
  override rates (0.20–0.50) and upgrade-probability spreads (0.50–0.95), the
  naive and pd-independent DGMs **never overlap**: naive gap ≥ 0.027 everywhere,
  pd-independent gap ≤ 0.006 everywhere.
- **Override features add almost nothing once the model is held fixed.** Under
  a single XGBoost model with identical hyperparameters, adding override
  features to the financial baseline changes ROC-AUC by **+0.0008**
  (0.9603 → 0.9611). The large ΔAUC reported in earlier drafts reflected a
  **logistic-vs-XGBoost model change**, not the override features.
- **Cross-validation folds are balanced** (default rate 0.0482 in every fold),
  but firm-level and temporal leakage **cannot be fully excluded** because the
  public dataset contains no firm identifier and `year_horizon` encodes a
  forecast horizon rather than a calendar year.

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
- **No firm identifier is available**, so records of the same firm across
  forecast horizons cannot be grouped. This is documented as a limitation.

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
│   ├── NB04_diag_correlation.ipynb          ← driver-variable diagnostic
│   ├── NB04b_override_simulation_independent.ipynb   ← three DGMs
│   ├── NB04c_override_simulation_replications.ipynb  ← 500-seed inference
│   ├── NB04d_override_simulation_sensitivity.ipynb   ← parameter grid
│   ├── NB05_statistical_model.ipynb
│   ├── NB06_ml_model.ipynb
│   ├── NB06b_validation_protocol.ipynb      ← leakage-aware validation
│   └── NB07_shap_explainability.ipynb
├── models/
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
| NB04_diag | Correlation diagnostic: find drivers near-orthogonal to `pd_system` |
| NB04b | Override simulation under three DGMs (naive / b1 / b2) |
| NB04c | Repeated simulations (500 seeds) with percentile confidence intervals |
| NB04d | Parameter sensitivity analysis over an override-rate × spread grid |
| NB05 | WoE / Information Value + statsmodels logit (per-DGM) |
| NB06 | XGBoost baseline model |
| NB06b | Leakage-aware validation: single-model feature-set comparison, fold balance, horizon-out check |
| NB07 | SHAP explainability (illustrative; interpreted as within-simulation only) |

**Run order:** NB01 → NB02 → NB03 → NB04_diag → NB04b → NB04c → NB04d →
NB05 → NB06 → NB06b → NB07

---

## Data-Generating Mechanisms

The three DGMs differ **only** in how per-firm upgrade probability is assigned;
everything downstream (magnitude, direction, `grade_diff`) is identical, so the
mechanisms are strictly comparable.

| DGM | Upgrade driver | Relation to `pd_system` | Role |
|-----|----------------|-------------------------|------|
| `naive` | `system_grade` (quantile of `pd_system`) | Strong (by construction) | Warning baseline — reproduces the circular design |
| `b1` | `Attr43` | Near-zero (\|ρ\| ≈ 0.03) | Risk-neutral upgrades; selection ≈ random w.r.t. risk |
| `b2` | `Attr29` (firm size) | Low (\|ρ\| ≈ 0.19), weak link to default | Risk-informed but pd-independent upgrades |

Outputs:
`data/processed/override_data_naive.parquet`, `…_b1.parquet`, `…_b2.parquet`

---

## Sign Convention

`grade_diff = final_ordinal − system_ordinal`, where a lower ordinal is a
better grade. Therefore:

- **negative `grade_diff` = upgrade**
- **positive `grade_diff` = downgrade**

This convention is applied consistently in the code and in the manuscript.

---

## Reproducibility

Random seeds are fixed throughout (`numpy.random.default_rng` with explicit
per-replication seeds; `random_state=42` for models). Running the notebooks in
order reproduces all reported tables and figures, including the 500-replication
intervals and the sensitivity grid.

---

## Note on pd_system

`pd_system` is included as a feature where appropriate, on the grounds that
evaluators observe the system score before overriding. However, this
repository explicitly **does not** rely on `pd_system` to claim override-feature
importance. NB06b reports a feature-set comparison **excluding** `pd_system`
precisely to isolate the incremental contribution of override features, which
is found to be negligible.

---

## Note on Analysis Level and Scope

All analyses are conducted at the **firm level** (unit: firm-year record) and
on **simulated** override decisions. The repository does **not** contain
evaluator-level identifiers, qualitative override reasoning, or institutional
context. Consequently:

- It **cannot** demonstrate that real evaluators overlook specific financial
  signals.
- SHAP results describe how the model fits the **simulated** data, not real
  evaluator behaviour.
- No deployment, evaluator-training, or committee-intervention claim is made.

The central methodological conclusion is that **valid empirical assessment of
override risk requires actual rating-override records**; simulation alone, once
design circularity is removed, cannot establish the effect.

---

## Citation

```bibtex
@article{anonymous2026override,
  title   = {[Title withheld for blind review]},
  author  = {Anonymous},
  journal = {[Journal withheld for blind review]},
  year    = {2026},
}
```

---

## License

This code is released for academic replication purposes.
The dataset is subject to the UCI ML Repository terms of use.
