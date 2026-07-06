# Survival Analysis & Mortality Risk Modeling — SUPPORT Cohort

Predicting time-to-death for seriously ill hospitalized patients using classical
survival analysis (Kaplan-Meier, Cox proportional hazards) alongside machine
learning baselines, with an emphasis on **rigorous evaluation, calibration, and
reproducibility** rather than raw accuracy.

## Why this project

A clinical prediction project built to demonstrate the modeling culture expected in
regulated healthcare / pharma settings: careful cohort definition, explicit handling
of censoring, honest baselines, calibration, explainability — and production-grade
engineering around the analysis (installable package, tests, CLI pipeline, CI).

## Engineering shape

Analysis lives in notebooks (the story); reusable logic lives in an installable
package (`src/support_survival/`) and is covered by tests. The full pipeline runs
headless from the command line — no Jupyter required.

```
support-survival/
├── src/support_survival/   # installable package
│   ├── data.py             # download + load SUPPORT
│   ├── features.py         # data-quality handling + feature engineering
│   ├── models.py           # logistic / gradient boosting / Cox
│   └── pipeline.py         # end-to-end, runnable via CLI
├── notebooks/              # analysis narrative (one per phase)
├── tests/                  # pytest suite
├── reports/                # figures + model card
├── scripts/                # helper scripts
├── .github/workflows/      # CI: lint + tests
└── pyproject.toml
```

## Quickstart

```bash
pip install -e ".[dev]"        # install package + dev tools
python -c "from support_survival import data; data.build_csv()"  # fetch data
pytest -q                      # run tests
python -m support_survival.pipeline --quick   # run the pipeline headless
jupyter lab notebooks/         # explore the analysis
```

Data are **never** committed (see `.gitignore`); results are reported in aggregate.

## Dataset

**SUPPORT** (~8,900 seriously ill patients, 5 US medical centers). `time` =
follow-up days, `event` = 1 death / 0 censored, plus 14 clinical covariates.
Canonical open survival dataset, no credentialing required.

## Roadmap

- [x] Phase 0 — Package scaffold, tests, CI, CLI pipeline
- [x] Phase 1 — Cohort definition & EDA
- [x] Phase 2 — Feature engineering & missing-data strategy
- [ ] Phase 3 — Baseline classifiers (logistic, gradient boosting) + calibration
- [ ] Phase 4 — Survival analysis (KM, Cox PH, PH-assumption checks, log-rank)
- [ ] Phase 5 — Validation, calibration curves, subgroup/fairness, SHAP
- [ ] Phase 6 — Model card finalized + FastAPI/Docker serving

See `reports/model_card.md` for the living methodology record.
