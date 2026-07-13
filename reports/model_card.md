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
follow-up: yes/no), on the Phase 2 feature set minus `race` (23 columns: 13
raw covariates, 4 missingness indicators, 6 derived clinical flags — see
"Excluding `race`" below), imputed inside each model's own pipeline and fit
with identical 5-fold stratified CV so the comparison is apples-to-apples:
- **Logistic regression** — median-imputed, scaled, no interaction terms.
  `class_weight="balanced"` was tested and **rejected**: it changed AUROC by
  <0.002 (0.644 → 0.643) but inflated predicted risk by up to 0.23 across
  probability bins (see Evaluation below) — not worth the calibration cost on
  this cohort's moderate 68%/32% split.
- **Gradient boosting** (XGBoost, 300 trees, depth 4, native missing-value
  handling — no imputation needed).

### Excluding `race` from every model
`race` is a numeric code (0–9) in this dataset's `.h5` export. Investigation
(prompted by wanting readable labels for the FastAPI demo form — see Phase 6):
the original SUPPORT/UCI source stores `sex` and `race` as **string** labels
("male"/"female"; "White, Black, Asian, Hispanic, other"), not numbers — so
DeepSurv's preprocessing must have invented its own numeric encoding when
converting them for their neural network, and that mapping is not published
anywhere we could find (same class of problem as the `wbc`/`serum_sodium`
swap in Phase 2). Our own data adds a second red flag specific to `race`: it
has **10** distinct codes, not the 5 the documentation describes, and the
most common single code covers only 33% of patients — implausible if any one
code were meant to be "White," typically 70–85% of a 1990s US cohort like
this one. `sex` has a commonly-used convention (0=female, 1=male) in other
derived copies of SUPPORT, but since even that dataset stores sex as strings,
we can't confirm it applies to this specific export either — `sex` is kept
as a model input (it's binary and directly interpretable as a code either
way) but never labeled "female"/"male" without a caveat.

**Decision:** `race` is excluded from every model's feature list
(`evaluate.EXCLUDED_FROM_FEATURES`) — Cox already excluded it in Phase 4 for
similar reasons; this extends the same call to the binary classifiers. Cost:
AUROC dropped by less than 0.002 for both models (see Evaluation below) —
removing an attribute we can't responsibly interpret was nearly free.
`race` stays in the DataFrame for subgroup reporting (Phase 5 could still
audit performance by `race` if its encoding were ever confirmed); it is just
never fed to a model as a predictor.

### Survival: Cox proportional hazards (Phase 4)
Covariates: `age`, `n_comorbidities`, `mean_bp`, `heart_rate`, `serum_creatinine`,
`wbc`, plus cancer status (see encoding decision below). Fit with `models.fit_cox`
(`notebooks/04_survival.ipynb`).

**`cancer` encoding.** Feeding `cancer` (0=none/other diagnosis, 1=present,
2=metastatic) to Cox as a single linear covariate produced a hazard ratio
below 1 (0.768) — i.e. "more cancer, less risk," which contradicts Phase 1's
log-rank result. Cause: `cancer` isn't monotonic in hazard in this cohort —
patients with *no* cancer have the **worst** raw survival of the three groups
(89.3% died, median 110 days), worse than metastatic cancer (75.9% died,
median 92 days), because SUPPORT's `cancer=0` bucket includes other
seriously-ill admission diagnoses (multi-organ failure with sepsis, coma,
acute respiratory failure) with high short-term mortality unrelated to
cancer. Non-metastatic cancer (`cancer=1`) has by far the best survival
(59.9% died, median 404 days). **Decision:** `models.encode_cancer_stage`
dummy-codes `cancer` (`cancer_present`, `cancer_metastatic`, reference
`cancer==0`) instead. This fixes the sign (HRs 0.479 and 0.743 respectively,
both directionally correct and consistent with the raw numbers) and improves
in-sample concordance (0.571 → 0.599).

**PH-assumption check** (`models.check_ph_assumption`, Schoenfeld-residual
test): on the dummy-coded model, every covariate except `n_comorbidities`
formally fails (p < 0.05) — expected at n=8,873, where the test is very
sensitive to even small deviations (lifelines' own documentation warns of
this). `cancer_present`/`cancer_metastatic` fail by a much larger margin
(p ≈ 1e-125/1e-12) than the vitals, and plausibly so: a cancer patient's
relative risk likely changes shape over a multi-year follow-up as disease
trajectories diverge, unlike a single blood-pressure reading's effect.
**Decision:** refit stratified by `cancer` (`strata=["cancer"]`) as a
robustness check — this gives cancer its own baseline hazard instead of a
shared proportional multiplier. Stratifying also resolved the PH assumption
for `age`, `heart_rate`, `n_comorbidities`, and `serum_creatinine` (all
p > 0.05 afterward), showing most of those "violations" were really cancer's
unmodeled non-linearity leaking into their residuals. `mean_bp` (p = 0.0011)
and `wbc` (p = 0.000075) still show a real, smaller violation — a disclosed
limitation, left for Phase 5 (binning, or a time-interaction term) rather
than fixed here.

**Which model is "the" result:** the dummy-coded model, reported as primary —
it keeps an interpretable hazard ratio for cancer status (a clinically
important variable), has higher concordance, and its PH violation for
`cancer` is disclosed and clinically plausible rather than hidden. The
stratified model is a robustness check confirming the other five hazard
ratios hold once cancer's confounding is removed structurally, not a
replacement.

**Log-rank cross-check** (`eda.plot_km_by_group`, reused from Phase 1,
`reports/km_by_cancer.png`): the three cancer-status groups differ with
p ≈ 7e-129 — independent, assumption-free confirmation that stratifying
`cancer` (rather than assuming one shared multiplicative effect) is
justified.

## Evaluation

### Phase 3 — binary mortality
5-fold stratified CV, same folds for both models (`notebooks/03_baselines.ipynb`,
`reports/roc_pr_curves.png`, `reports/calibration_curves.png`):

| Model | AUROC | Average Precision |
|---|---|---|
| Logistic regression | 0.643 | 0.784 |
| Gradient boosting | 0.698 | 0.822 |

- **Discrimination:** gradient boosting has a real, consistent edge across the
  full ROC and PR curves, not just the summary numbers — reported honestly
  rather than only showing the better-looking model.
- **Calibration** (`reports/calibration_curves.png`): both models track the
  diagonal reasonably well — gradient boosting within about 4–5 points at
  every bin, logistic regression within about 5 points too. (An earlier
  version of this entry claimed logistic regression "systematically
  underestimates risk by 0.13–0.23" — that number was the **rejected**
  `class_weight="balanced"` configuration's miscalibration, mistakenly
  attributed to the model actually shipped. Caught while re-verifying
  calibration for the `race` exclusion above; the plot was always correct,
  only this description was wrong.)
- **Explainability** (`reports/shap_summary.png`, gradient boosting fit on the
  full cohort — for interpretation, not held-out evaluation): `cancer` is by
  far the strongest driver of predicted risk (mean |SHAP| ≈ 0.54), then `age`
  (≈ 0.30), then `serum_creatinine`, `wbc`, `mean_bp`, `temperature`, `sex`,
  `heart_rate` with more modest, comparable contributions (filling the ranks
  `race` occupied before it was excluded). This matches the Phase 1 finding
  that cancer status splits the Kaplan-Meier curves apart with overwhelming
  significance — the binary and survival views agree.

### Phase 4 — survival (C-index)
`models.cross_val_concordance` mirrors Phase 3's out-of-fold rigor (a plain
`.concordance_index_` is in-sample and optimistic):

| Model | CV concordance |
|---|---|
| Cox, dummy-coded cancer | 0.598 |
| Cox, stratified by cancer | 0.566 |

Compared to Phase 3: both Cox variants trail gradient boosting's AUROC
(0.698); the dummy-coded Cox is close to logistic regression's AUROC (0.643).
The stratified model's lower score is largely a metric artifact, not
necessarily worse real-world utility: concordance is computed from the risk
score alone, which structurally excludes `cancer` once it's a stratum rather
than a covariate — so the model that most honestly respects the PH
assumption is penalized by a metric that can't see its most informative
variable. Neither Cox variant "wins" on discrimination; the value of Phase 4
is the interpretable hazard ratios and disclosed assumption-checking, not a
higher score.

### Phase 5 — held-out validation, calibration, uncertainty, fairness

Phase 3/4 compared candidate models via cross-validation on the full cohort —
appropriate for model *selection*, but every CV fold still touches every
patient eventually. Phase 5 reports final numbers for the already-chosen
models on a three-way, stratified-by-event split created once and never
revisited (`validate.split_train_val_test`, `notebooks/05_validation.ipynb`):
train 5,323 / validation 1,775 / test 1,775 (60/20/20), event rate ~68% in
all three. `test` is used only for final reporting below — never to fit a
model, tune a hyperparameter, or fit a recalibrator (that used `val`).

**Calibration & recalibration.** Gradient boosting (trained on `train` only,
same configuration Phase 3 selected via CV, on the `race`-excluded feature
set — no new tuning here) scores Brier 0.1939 on `test`. An isotonic
recalibrator fit strictly on `val` (`validate.fit_isotonic_recalibrator`)
barely changes it (0.1944, very slightly *worse*) — consistent with Phase
3's finding that this model was already reasonably calibrated; recalibration
was checked, not skipped, and simply wasn't needed. Reliability diagram:
`reports/calibration_test_before_after.png`. **Decision:** serve the raw
(non-recalibrated) model.

**Uncertainty (bootstrap, 2,000/1,000 resamples of `test` only):**

| Metric | Point | 95% CI |
|---|---|---|
| Gradient boosting AUROC | 0.700 | [0.674, 0.725] |
| Gradient boosting Brier | 0.194 | [0.185, 0.203] |
| Cox concordance (dummy-coded) | 0.608 | [0.590, 0.626] |

Both point estimates land close to their Phase 3/4 cross-validated
counterparts (0.698, 0.598) — good agreement between the CV comparison and
this independent held-out estimate. Neither CI is wide enough to change the
qualitative conclusions from Phase 3/4, and neither approaches the 0.85+
range that would call for a leakage investigation.

**Subgroup performance & calibration** (AUROC = discrimination, Brier =
calibration; `test` set, raw model):

| Age band | n | Event rate | AUROC | Brier |
|---|---|---|---|---|
| <50 | 363 | 49.9% | 0.744 | 0.205 |
| 50–64 | 524 | 71.9% | 0.661 | 0.194 |
| 65–74 | 465 | 71.0% | 0.648 | 0.195 |
| 75–84 | 338 | 76.0% | 0.599 | 0.180 |
| **85+** | 85 | 72.9% | **0.595** | 0.199 |

| Sex (code) | n | Event rate | AUROC | Brier |
|---|---|---|---|---|
| 0 | 997 | 69.4% | 0.717 | 0.186 |
| 1 | 778 | 66.2% | 0.680 | 0.204 |

**Documented limitation:** discrimination degrades with age — AUROC 0.744
(<50) down to **0.595 for patients 85+**, a real ~15-point gap, even though
Brier score stays roughly flat (~0.18–0.20) across bands. Calibration and
discrimination are different failure modes: this model is passably
calibrated *on average* for the oldest patients while ranking them
noticeably worse than younger patients. **Any future use of this model
should flag predictions for patients 85+ as lower-confidence.** By sex: a
smaller but present gap (AUROC 0.717 vs 0.680). `sex` is documented only as
"encoded" (`data.DESCRIPTIONS`) — which code is male vs female is not
independently confirmed by the data or its source documentation (see
"Excluding `race`" above — the same investigation covered `sex`), so this is
reported by code value without asserting which is which.

**Error analysis.** The model's largest errors are clinically coherent, not
random. Over-predicted (high predicted risk, but survived): 12 of the 15
worst cases have `cancer=0`, vastly overrepresented vs. the 19.7% base rate
— `cancer=0` is this cohort's highest-mortality group (other severe SUPPORT
diagnoses, not "healthy"; see Phase 4), so the model has learned a strong,
usually-correct "cancer=0 → high risk" prior that fails on the minority who
survive anyway. Under-predicted (low predicted risk, but died): dominated by
young (20s–50s) `cancer=1` patients with 0–2 comorbidities — this cohort's
best-prognosis profile, and the mirror-image failure of the same prior.

## Serving (Phase 6)

`src/support_survival/api.py`, a FastAPI app: `POST /predict` (Pydantic-validated
input, returns `{risk_probability, model_version}`), `GET /health` (container
health check), `GET /` (a small interactive demo page: form → prediction →
a risk meter with a marker at the cohort's 68% baseline event rate, and a
plain-language risk band — illustrative only, not a validated clinical risk
category). `scripts/train_and_save.py` trains gradient boosting on the
**full** cohort (not held back — Phase 5 already produced an honest, held-out
performance estimate for this architecture: AUROC 0.700, 95% CI
[0.674, 0.725]) and persists it to `models/gradient_boosting.joblib`, which is
gitignored and regenerated from scratch, not committed.

**Train/serve consistency:** `predict()` calls `features.build_features`
directly — the same function, same import, used in every training notebook.
No preprocessing logic is duplicated in the app. The Pydantic schema asks for
13 of the 14 raw covariates — `race` is omitted entirely (not just unlabeled)
per the exclusion decision above, and `sex`'s field description flags that
its 0/1 mapping to male/female isn't independently confirmed for this
dataset. The demo page's form mirrors this: no `race` input, and the `sex`
field is labeled "code; mapping unconfirmed" rather than "Male"/"Female".

**Input validation:** the Pydantic schema rejects genuinely impossible values
(negative age, out-of-range vitals, wrong types, missing fields) with a 422
before a request ever reaches the model. It deliberately does **not** reject
physiologically-implausible-but-representable sentinels (e.g. `mean_bp == 0`)
at the API boundary — `features.build_features` is responsible for handling
those exactly as it does in training (converting them to `NaN`), and letting
the schema reject them instead would silently diverge serving's behavior from
training's.

**Bug found and fixed during Phase 6 verification:** the model path was
originally computed as `Path(__file__).resolve().parents[2]` — this works
under the `pip install -e .` used everywhere in local dev (where `__file__`
still resolves inside the source tree), but breaks under a normal, non-editable
install (`pip install .`, used in the Dockerfile): the file lands in
`site-packages`, and counting parent directories no longer reaches the repo
root. This was **not caught by any test** — every local test ran under the
editable install and never exercised the failure mode. It only surfaced when
the built Docker image was actually run and `POST /predict` returned a 500.
Fixed by resolving the model path relative to the working directory (which
both local dev and the container's `WORKDIR /app` set correctly), with an
environment-variable override. Verified with a full rebuild + container run:
`/health` → `{"status":"ok"}`, `/predict` → a valid risk probability,
malformed input → 422.

**Follow-up (fixed):** `data.py`'s `ROOT`/`DATA_DIR` used the same
`Path(__file__).resolve().parents[2]` pattern that caused the bug above —
flagged here at the time as a known adjacent risk, not yet fixed. It never
crashed (it wrote and read the downloaded dataset from a self-consistent but
semantically wrong location inside `site-packages` instead of a proper data
directory, so reads always matched writes), and it's on the live path the
triage panel actually exercises (`GET /triage` calls `data.load()`). Now
resolved the same way as `MODEL_PATH`: relative to the working directory,
with a `SUPPORT_SURVIVAL_ROOT` env var override, and a regression test
(`test_root_is_cwd_relative_not_file_relative`) guarding against `__file__`
creeping back in.

### Triage panel (`GET /triage`, `GET /triage-view`)

Reframed from an earlier, rejected idea (estimating freed-up hospital beds
from predicted mortality — a poor fit: dying patients can occupy a bed
*longer* than quickly-discharged low-risk ones, and SUPPORT has no
discharge/transfer timestamps to model that anyway). The reframe: a
**risk-stratified patient panel for care-team triage**, closer to what
SUPPORT was originally built for (improving prognosis communication for
seriously ill patients — literally what the acronym stands for).

**Why Cox, not just the binary classifier.** A single risk probability
conflates "high risk very soon" with "high risk eventually" — not useful for
urgency. `models.predict_risk_at_horizons` uses the Phase 4 Cox model's
individual survival curve to give each patient a 30-day and 90-day mortality
risk, not just one number. Tiers: **Urgent** (30-day risk ≥ 40%, roughly the
cohort's 90th percentile), **Monitor** (overall risk ≥ 65% but not urgent),
**Routine** (otherwise). `train_and_save.py` now also fits and persists the
Phase 4 primary Cox model (dummy-coded cancer, full cohort) alongside the
classifier, in the same artifact.

Patients come from a **fixed random sample of the Phase 5 held-out test
set** (`random_state=7`, stable across requests) — there is no live hospital
feed; this is a demonstration.

**A second train/serve bug, same family as the path bug above, found while
building this.** The Cox model was trained (Phase 4, and again in
`train_and_save.py`) on **raw** covariates — `flag_implausible` was never
applied to its inputs. The first version of `/triage` built its Cox input
from `features.build_features(df)`, which *does* flag implausible zeros
(`mean_bp`, `heart_rate`, `wbc` → `NaN`) — so a handful of patients silently
got `null` 30-/90-day risk (`predict_survival_function` can't handle `NaN`
inputs). Fixed by splitting on the **raw** `df` for the Cox input (matching
what the model actually trained on) while still using the featurized frame
for the binary classifier — the same patients either way, since `df` and
`feat` share an index; just the right preprocessing state for each model.
Caught before shipping, by checking the actual endpoint output, not just
that it returned 200.

**Explicit, undismissable limitation:** the panel's least reliable
predictions are for patients 85+ (Phase 5: AUROC 0.595 vs 0.744 for
under-50s) — precisely the group where a triage/goals-of-care call is most
consequential. The triage page states this in a banner above the table, not
buried in a caveats section. Tiers are a prompt for a conversation, never an
automated allocation decision — the same caution that motivated rejecting
the original "beds freed by predicted death" framing applies here too, just
less severely.

## Phase 7 — Follow-up on two deferred items

Two things were left explicitly unfinished earlier: Phase 3's "no
hyperparameter search backed these defaults," and Phase 4's disclosed
`mean_bp`/`wbc` PH violations, "left for Phase 5... rather than fixed here"
(Phase 5 never did it either). `notebooks/07_followups.ipynb` closes both
out — one becomes a real, adopted improvement; the other stays a disclosed,
partial result rather than a forced fix.

### Hyperparameter search (adopted)
`models.tune_gradient_boosting` runs a 40-sample randomized search over tree
count, depth, learning rate, and both subsampling rates, scored by AUROC
under the same 5-fold stratified CV protocol Phase 3 used to compare models.
Best CV AUROC: 0.708. Verified on Phase 5's untouched held-out test set
(never used by the search): **AUROC 0.700 → 0.711, Brier 0.194 → 0.192** —
both discrimination and calibration improved together, not a trade-off.
Winning configuration: `n_estimators=120`, `max_depth=4`,
`learning_rate≈0.019`, `subsample≈0.78`, `colsample_bytree≈0.84`,
`min_child_weight=7` — fewer, shallower trees with a lower learning rate
than Phase 3's hand-set defaults (300 trees, `learning_rate=0.05`).
**Decision:** `models.gradient_boosting_tuned()` is now what
`scripts/train_and_save.py` ships and the triage panel serves. Phase 3's own
reported numbers (0.643/0.698 AUROC) are left unchanged: they describe the
untuned baseline that was actually compared at the time and stay
reproducible from `models.gradient_boosting()`.

### Cox PH violations: `mean_bp` fixed, `wbc` not (disclosed, not forced)
Standard remedy for a covariate whose continuous form violates proportional
hazards: model it categorically instead. Both `mean_bp` and `wbc` were split
into three clinical bands (low/normal/high; implausible zeros excluded, 96
of 8,873 rows). Result: **`mean_bp` resolved** (`mean_bp_low` p≈0.31,
`mean_bp_high` p≈0.19, both comfortably clear the 0.05 threshold that the
continuous version failed at p≈0.001). **`wbc` did not** (`wbc_high`
borderline at p≈0.044, `wbc_low` still fails at p≈0.000002) — banding didn't
fix it. A third covariate, `heart_rate`, newly showed a violation (p≈0.003)
not present in the original check, reported rather than dropped quietly;
plausibly the 96 excluded rows shift the sample slightly. Concordance was
essentially unchanged (0.585 vs 0.583 for this vitals-only comparison) — no
gain, no loss.

Plausible reason `wbc` resists this fix where `mean_bp` doesn't: a single
low blood-pressure reading likely reflects noise around a stable
physiological state, while an abnormal white blood cell count (especially
low, e.g. chemotherapy-induced neutropenia) can mark a changing
*trajectory* — its relationship to risk plausibly shifts shape over
follow-up in a way three static bands can't capture. Properly fixing that
needs a genuine time-varying-covariate model (`CoxTimeVaryingFitter`,
restructuring each patient's follow-up into time-sliced episodes) — a
materially larger undertaking than this follow-up, out of scope here.
**Decision:** Phase 4's primary served Cox model is unchanged. Adopting the
banded version would fix one violation, surface another, and not fix the
one it targeted, for no concordance gain — not a clean win, and not worth
touching an already-validated model for.

### `data.py` path bug (fixed)
See the Phase 6 entry above: `ROOT`/`DATA_DIR` no longer derive from
`__file__`, closing the adjacent risk flagged there at the time.

## Ethical & governance notes
- No raw patient-level data committed.
- Results reported in aggregate only.
- Dataset used under its open-access terms.
