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
  (32.0%); median follow-up (all patients, including censored) 231 days, max
  2,029 days. Median time-to-death **among the 6,036 observed deaths only** is
  57 days — this matches the 68.10%-event-rate / 58-day figure reported for
  this exact dataset in the original DeepSurv paper (Katzman et al., 2016,
  arXiv:1606.00931), which we used as an independent check that `time`/`event`
  are read correctly (see below). See `notebooks/01_eda.ipynb` for the
  derivation.
- **Overall Kaplan-Meier:** median survival time 231 days
  (`reports/km_overall.png`).
- **Stratified check (cancer status):** three levels (none / present /
  metastatic). Kaplan-Meier curves separate clearly and a multivariate
  log-rank test rejects equal survival across groups (p ≈ 7e-129,
  `reports/km_by_cancer.png`) — consistent with clinical expectation and a
  reason to keep `cancer` as a Cox covariate in Phase 4.

## Preprocessing & data-quality decisions

### `wbc` / `serum_sodium` column swap (Phase 2)
Phase 2 EDA checked the actual value ranges of all 14 covariates against
clinical reference ranges (not done in Phase 1, which only checked zero
counts). Two columns didn't match their labels:

| Column (as shipped) | Observed range / median | Expected for its label | Matches instead |
|---|---|---|---|
| `wbc` | median 137, range 110–181 (narrow) | WBC ~4–11 (x1000/mm³) — **no match** | Serum sodium (135–145 mEq/L, wider in critically ill patients) |
| `serum_sodium` | median 10.6, range 0–200 (right-skewed) | Sodium 135–145 mEq/L — **no match** | WBC (right-skewed, typical median ~9–11 x1000/mm³) |

Evidence the two are swapped, not just noisy:
1. The ranges match textbook values for the *other* variable, in both cases.
2. The Vanderbilt/Harrell SUPPORT codebook's default imputation value for
   `wblc` is 9 (x1000/mm³) — matches the median (10.6) of the column labeled
   `serum_sodium` almost exactly.
3. The DeepSurv paper's reported event rate (68.10%) and median death time
   (58 days) for this exact dataset match ours (68.03%, 57 days) almost
   exactly, confirming `time`/`event` — and by extension the rest of the
   column-reading logic — are correct; the mislabeling looks specific to
   these two columns in the shipped `.h5` file, not a general reading bug.

No definitive source script was found that proves this beyond doubt (the
`.h5` file has no embedded column metadata, and the original preprocessing
script is not published), so this is a documented, evidence-based inference,
not a certainty. **Decision:** swap the two labels in
`data.FEATURES` (see the `NOTE` comment there) so `wbc` and `serum_sodium`
hold the clinically plausible values going forward. A regression test
(`test_wbc_and_serum_sodium_are_in_clinically_plausible_ranges`) guards
against this being silently reverted.

### Implausible-zero handling
Zero is physiologically impossible for four vitals, so `features.flag_implausible`
replaces `0` with `NaN` for each (documented inline in `IMPLAUSIBLE_ZERO_COLS`):
- `mean_bp == 0` (51 patients, 0.6%) — Phase 0 decision, confirmed in EDA.
- `heart_rate == 0` (82, 0.9%), `resp_rate == 0` (65, 0.7%) — found in Phase 1 EDA.
- `wbc == 0` (10, 0.1%; this is the column formerly mislabeled `serum_sodium` —
  see the swap above) — a live patient cannot have zero white blood cells.

Zeros in `sex`, `race`, `n_comorbidities`, `diabetes`, `dementia`, `cancer` are
legitimate values and are left untouched.

### Missingness indicators
`features.add_missingness_indicators` adds a `{col}_missing` binary flag for
each of the four columns above, computed *before* any imputation. Rationale:
whether a vital sign was recorded may itself carry signal (e.g. differences in
monitoring protocol across the 5 SUPPORT hospitals, or how unstable a patient
was) — collapsing that into an imputed value would throw the signal away.

Actual numeric imputation is **not** done in `features.py`: it happens inside
the Phase 3+ model pipelines (`sklearn` `Pipeline`/`ColumnTransformer`), fit on
training folds only, so imputation statistics never leak from validation/test
data into training.

### Derived clinical features
All thresholds are standard clinical cutoffs, not fitted statistics:
- `age_over_70` — age ≥ 70.
- `creatinine_high` — serum creatinine > 1.5 mg/dL (reduced renal function).
- `tachycardic` — heart rate > 100 bpm.
- `hypotensive` — mean arterial pressure < 65 mmHg (the threshold used in
  sepsis guidelines below which organ perfusion is at risk).
- `leukocytosis` — white blood cell count > 11 (x1000/mm³; normal upper limit).
- `sodium_abnormal` — serum sodium outside the normal range (135–145 mEq/L),
  flagging hypo- or hypernatremia in either direction.

## Models

### Binary mortality baselines (Phase 3)
Two models, compared on the **binary** view of the outcome (died during
follow-up: yes/no), on the full Phase 2 feature set (24 columns: 14 raw
covariates, 4 missingness indicators, 6 derived clinical flags), imputed
inside each model's own pipeline and fit with identical 5-fold stratified CV
so the comparison is apples-to-apples:
- **Logistic regression** — median-imputed, scaled, no interaction terms.
  `class_weight="balanced"` was tested and **rejected**: it changed AUROC by
  <0.002 (0.647 → 0.646) but inflated predicted risk by 0.13–0.23 across
  probability bins (see Evaluation below) — not worth the calibration cost on
  this cohort's moderate 68%/32% split.
- **Gradient boosting** (XGBoost, 300 trees, depth 4, native missing-value
  handling — no imputation needed).

Survival (time-to-event) modeling — Kaplan-Meier, Cox PH — is Phase 4.

## Evaluation

### Phase 3 — binary mortality
5-fold stratified CV, same folds for both models (`notebooks/03_baselines.ipynb`,
`reports/roc_pr_curves.png`, `reports/calibration_curves.png`):

| Model | AUROC | Average Precision |
|---|---|---|
| Logistic regression | 0.646 | 0.785 |
| Gradient boosting | 0.700 | 0.824 |

- **Discrimination:** gradient boosting has a real, consistent edge across the
  full ROC and PR curves, not just the summary numbers — reported honestly
  rather than only showing the better-looking model.
- **Calibration** (`reports/calibration_curves.png`): gradient boosting tracks
  the diagonal closely (within a few points at every bin). Logistic
  regression systematically **underestimates** risk by 0.13–0.23 across bins
  — this is what drove the decision to drop `class_weight="balanced"` above;
  with it, the miscalibration was even worse.
- **Explainability** (`reports/shap_summary.png`, gradient boosting fit on the
  full cohort — for interpretation, not held-out evaluation): `cancer` is by
  far the strongest driver of predicted risk (mean |SHAP| ≈ 0.53), then `age`
  (≈ 0.30), then `race`, `serum_creatinine`, `temperature`, `wbc`, `mean_bp`,
  `heart_rate` with more modest, comparable contributions. This matches the
  Phase 1 finding that cancer status splits the Kaplan-Meier curves apart with
  overwhelming significance — the binary and survival views agree.

### Phase 5 (not yet done)
- Discrimination: C-index (survival).
- Uncertainty: bootstrap confidence intervals.
- Subgroup / fairness analysis (age band, sex).

## Ethical & governance notes
- No raw patient-level data committed.
- Results reported in aggregate only.
- Dataset used under its open-access terms.
