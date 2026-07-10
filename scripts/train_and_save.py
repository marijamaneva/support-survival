"""Train the final gradient-boosting mortality model and persist it for serving.

Run this to (re)generate the model artifact from scratch:

    python scripts/train_and_save.py

The artifact is never committed (see .gitignore) -- this script is how you
get it back, deterministically, from the same code the rest of the project
uses. Trains on the full cohort rather than holding data back: Phase 5 already
produced an honest, held-out performance estimate for this model architecture
(AUROC 0.699, 95% CI [0.672, 0.724]), so the artifact shipped for serving uses
all available data instead of sacrificing some of it a second time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib

from support_survival import data, evaluate, features, models

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "gradient_boosting.joblib"


def main() -> Path:
    df = data.load()
    feat = features.build_features(df)
    feature_columns = evaluate.feature_columns(feat)

    gb = models.gradient_boosting()
    gb.fit(feat[feature_columns], feat["event"])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": gb,
            "feature_columns": feature_columns,
            "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    main()
