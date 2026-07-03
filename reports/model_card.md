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
_Phase 1: inclusion/exclusion criteria and how the analysis sample was derived._

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
