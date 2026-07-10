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
│   ├── eda.py              # exploratory-analysis plots and summaries
│   ├── models.py           # logistic / gradient boosting / Cox construction
│   ├── evaluate.py         # CV comparison, ROC/PR, calibration, SHAP
│   ├── validate.py         # held-out split, bootstrap CI, subgroup fairness
│   ├── pipeline.py         # end-to-end, runnable via CLI
│   └── api.py              # FastAPI serving layer
├── notebooks/              # analysis narrative (one per phase)
├── tests/                  # pytest suite
├── reports/                # figures + model card
├── scripts/                # helper scripts: sanity_checks.py, train_and_save.py
├── models/                 # trained artifact (gitignored, regenerated on demand)
├── .github/workflows/      # CI: lint + tests
├── Dockerfile
└── pyproject.toml
```

## Guard-rails: `scripts/sanity_checks.py`

A permanent, mechanical version of the checks a careful reviewer would run by
hand before trusting any number in this repo — run manually or as part of
`pytest -q` (via `tests/test_sanity_checks.py`):

- the overall Kaplan-Meier curve is non-increasing and starts at 1.0;
- `time`/`event` never leak into the predictive feature list, for either the
  binary classifiers or the Cox model;
- the Cox model's cross-validated concordance lands in a plausible range
  (0.5-0.85) — outside it usually means either a useless model or leakage;
- the binary classifiers' AUROC stays below 0.90 — this dataset's mortality
  signal doesn't support anything higher, so a value at or above that is
  treated as a probable leakage bug, not a good result.

```bash
python scripts/sanity_checks.py
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

## Serving

The mortality-risk model is served behind a small FastAPI app
(`src/support_survival/api.py`). It reuses `features.build_features` directly
from the training package — no preprocessing is reimplemented in the app, so
training and serving can never silently drift apart.

```bash
python scripts/train_and_save.py       # trains on the full cohort, saves models/gradient_boosting.joblib
uvicorn support_survival.api:app --reload   # http://localhost:8000/docs
```

The trained model is **not** committed (see `.gitignore`) — `train_and_save.py`
regenerates it from scratch, deterministically, from the same code the rest
of the project uses. `POST /predict` returns `{"risk_probability", "model_version"}`,
so every prediction is traceable to the artifact that produced it; malformed
input (missing field, wrong type, out-of-range value) is rejected with a 422
by the Pydantic schema before it ever reaches the model. `GET /health` is
used as the container health check.

### Docker

```bash
docker build -t support-survival .     # trains the model at build time (needs network access)
docker run -p 8000:8000 support-survival
curl http://localhost:8000/health
```

## Dataset

**SUPPORT** (~8,900 seriously ill patients, 5 US medical centers). `time` =
follow-up days, `event` = 1 death / 0 censored, plus 14 clinical covariates.
Canonical open survival dataset, no credentialing required.

## Roadmap

- [x] Phase 0 — Package scaffold, tests, CI, CLI pipeline
- [x] Phase 1 — Cohort definition & EDA
- [x] Phase 2 — Feature engineering & missing-data strategy
- [x] Phase 3 — Baseline classifiers (logistic, gradient boosting) + calibration
- [x] Phase 4 — Survival analysis (KM, Cox PH, PH-assumption checks, log-rank)
- [x] Phase 5 — Validation, calibration curves, subgroup/fairness, SHAP
- [x] Phase 6 — Model card finalized + FastAPI/Docker serving

See `reports/model_card.md` for the living methodology record.
