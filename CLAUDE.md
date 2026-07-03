# Project context for Claude Code

This file orients you (Claude Code) on this repository. Read it before making changes.

## What this project is

A clinical survival-analysis portfolio project on the **SUPPORT** cohort (~8,900
seriously ill patients; `time` = follow-up days, `event` = 1 death / 0 censored,
14 clinical covariates). The audience is Swiss pharma / medtech hiring managers, so
the priorities are: methodological rigor, correct handling of censoring, honest
baselines, calibration, explainability, and clean engineering.

## Non-negotiable principles

1. **Censoring is real.** This is survival analysis, not plain binary
   classification. Never treat censored patients as "alive forever." Use
   lifelines (Kaplan-Meier, Cox) for the survival view.
2. **No data leakage.** Any imputation/scaling must be learned on training folds
   only (inside sklearn Pipelines / cross-validation), never on the full dataset.
3. **Reusable logic goes in `src/support_survival/`, not in notebooks.** Notebooks
   import from the package and tell the analysis story. If you write a function
   worth reusing, put it in the package and add a test.
4. **Every non-trivial function gets a test** in `tests/`. Keep `pytest -q` green.
5. **Document decisions, not just code.** Update `reports/model_card.md` whenever a
   modeling or data-cleaning decision is made, explaining *why*.
6. **Never commit data.** `data/` and `models/` are gitignored. Report results in
   aggregate only.

## How to work

- Install: `pip install -e ".[dev]"`
- Get data: `python -c "from support_survival import data; data.build_csv()"`
- Test: `pytest -q`  — must stay green after every change.
- Lint: `ruff check src tests`
- Run pipeline headless: `python -m support_survival.pipeline --quick`

## Existing modules

- `data.py` — download/load SUPPORT; `FEATURES`, `DESCRIPTIONS` constants.
- `features.py` — `flag_implausible`, `add_clinical_features`, `build_features`.
- `models.py` — `logistic_baseline`, `gradient_boosting`, `fit_cox`.
- `pipeline.py` — end-to-end CLI entry point.

## The phased roadmap (work one phase at a time)

Each phase = new/updated notebook in `notebooks/` + supporting code in the package
+ tests + a model-card update. Ask me before jumping ahead a phase.

- **Phase 1 — EDA:** `notebooks/01_eda.ipynb`. Outcome/censoring overview, overall
  Kaplan-Meier, feature distributions, data-quality scan (flag `mean_bp==0` and
  other implausible values), stratified KM by a clinical factor. Save figures to
  `reports/`.
- **Phase 2 — Features & missing data:** expand `features.py` with justified
  cleaning + derived features; tests for each; document in model card.
- **Phase 3 — Baselines + calibration:** logistic vs gradient boosting for binary
  mortality; proper CV; ROC/PR; **calibration curves**; SHAP for the boosting model.
- **Phase 4 — Survival analysis:** Cox PH with PH-assumption checks
  (`check_assumptions`), log-rank tests between strata, hazard-ratio
  interpretation, C-index; compare against the binary view.
- **Phase 5 — Validation & fairness:** bootstrap CIs, subgroup performance (age
  band, sex), calibration by subgroup, error analysis.
- **Phase 6 — Serving:** a small FastAPI app exposing a `/predict` endpoint that
  loads a trained model and returns a risk score; a Dockerfile; a smoke test.

## Style

- Python 3.10+, type hints, docstrings that explain intent.
- Keep notebooks lean: narrative + calls into the package, not walls of logic.
- Prefer clarity over cleverness. A reviewer should follow every decision.
