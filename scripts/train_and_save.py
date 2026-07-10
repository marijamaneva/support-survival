"""Train the final gradient-boosting mortality model and persist it for serving.

Run this to (re)generate the model artifact from scratch:

    python scripts/train_and_save.py

The artifact is never committed (see .gitignore) -- this script is how you
get it back, deterministically, from the same code the rest of the project
uses. Trains on the full cohort rather than holding data back: Phase 5 already
produced an honest, held-out performance estimate for this model architecture
(AUROC 0.700, 95% CI [0.674, 0.725]), so the artifact shipped for serving uses
all available data instead of sacrificing some of it a second time.

Also fits the primary Cox model from Phase 4 (dummy-coded cancer) on the full
cohort, saved alongside the classifier -- the triage view (`GET /triage`)
needs its individual survival curves for 30/90-day risk, which a binary
classifier can't provide.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib

from support_survival import data, evaluate, features, models

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "gradient_boosting.joblib"


COX_COVARIATES = [
    "age", "n_comorbidities", "mean_bp", "heart_rate", "serum_creatinine", "wbc",
    "cancer_present", "cancer_metastatic",
]


def main() -> Path:
    df = data.load()
    feat = features.build_features(df)
    feature_columns = evaluate.feature_columns(feat)

    gb = models.gradient_boosting()
    gb.fit(feat[feature_columns], feat["event"])

    # Same primary Cox model chosen in Phase 4 (dummy-coded cancer, not
    # stratified) -- used for the triage view's 30/90-day risk, which needs
    # the individual survival curve the binary classifier can't provide.
    df_cox = models.encode_cancer_stage(df)
    cph = models.fit_cox(df_cox, COX_COVARIATES)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": gb,
            "feature_columns": feature_columns,
            "cox_model": cph,
            "cox_covariates": COX_COVARIATES,
            "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    main()
