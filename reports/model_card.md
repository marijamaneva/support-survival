# Model Card — SUPPORT Mortality / Survival Model

> A living document, filled in as the project progresses. Modeled on the
> "Model Cards for Model Reporting" framework, because clinical and pharma
> employers think in terms of documented, auditable models.

## Intended use
- **Purpose:** educational / portfolio demonstration of survival modeling. **Not**
  for clinical use.
- **Out of scope:** any real patient-facing decision.

## Data
- **Source:** SUPPORT study (~8,900 seriously ill patients, 5 US medical centers).
- **Outcome:** time-to-death (`time`, days) with right-censoring (`event`).
- **Event rate:** ~68% observed deaths, ~32% censored.
- **Limitations:** 1990s US cohort; not representative of current Swiss/EU care.

## Cohort definition
- **Sample:** the full SUPPORT extract as shipped by the DeepSurv preprocessing —
  8,873 patients, 14 covariates, `time` + `event`. No records excluded; the
  DeepSurv authors' own inclusion/exclusion criteria are inherited as-is (this
  project does not re-derive the cohort from the original SUPPORT trial data).
- **Outcome overview:** 6,036 deaths observed (68.0%), 2,837 right-censored
  (32.0%); median follow-up 231 days, max 2,029 days. See
  `notebooks/01_eda.ipynb` for the derivation.
- **Overall Kaplan-Meier:** median survival time 231 days
  (`reports/km_overall.png`).
- **Stratified check (cancer status):** three levels (none / present /
  metastatic). Kaplan-Meier curves separate clearly and a multivariate
  log-rank test rejects equal survival across groups (p ≈ 7e-129,
  `reports/km_by_cancer.png`) — consistent with clinical expectation and a
  reason to keep `cancer` as a Cox covariate in Phase 4.
- **Data-quality scan (zero values):** `mean_bp == 0` (51 patients, 0.6%) is
  already treated as missing in `features.flag_implausible`. EDA surfaced
  further physiologically-implausible zeros not yet handled:
  `heart_rate == 0` (82, 0.9%), `resp_rate == 0` (65, 0.7%), and
  `serum_sodium == 0` (10, 0.1%). These are candidates for
  `IMPLAUSIBLE_ZERO_COLS` in Phase 2, not fixed here. Zeros in `sex`, `race`,
  `n_comorbidities`, `diabetes`, `dementia`, `cancer` are legitimate values
  and require no action.

## Preprocessing & data-quality decisions
_Phase 2. Record each decision, e.g.:_
- `mean_bp == 0` treated as missing (impossible for a live patient).
- Outlier handling for `wbc`, `serum_creatinine`, `temperature`.
- Imputation learned on training folds only (no leakage).

## Models
_Phases 3–4:_
- Baseline: logistic regression, gradient boosting (binary mortality).
- Survival: Kaplan-Meier, Cox proportional hazards (+ PH checks).

## Evaluation
_Phase 5:_
- Discrimination: C-index (survival), AUROC / AUPRC (binary).
- **Calibration:** calibration curves — a predicted 20% risk should mean ~20%.
- Uncertainty: bootstrap confidence intervals.
- Subgroup / fairness analysis (age band, sex).

## Ethical & governance notes
- No raw patient-level data committed.
- Results reported in aggregate only.
- Dataset used under its open-access terms.
