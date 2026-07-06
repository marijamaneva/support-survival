"""End-to-end pipeline runnable from the command line.

    python -m support_survival.pipeline
    # or, after `pip install -e .`:
    support-pipeline

This deliberately does NOT require Jupyter: it demonstrates the project runs
headless, which is what a CI job or a reviewer cloning the repo will do first.
Notebooks tell the analysis story; this script proves the code executes.
"""
from __future__ import annotations

import argparse

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from support_survival import data, features, models


def run(quick: bool = False) -> dict:
    df = data.load()
    feat = features.build_features(df)

    numeric_cols = [c for c in data.FEATURES]  # imputer handles the NaNs we flagged
    # Binary target: did the patient die during observed follow-up?
    y = df["event"].to_numpy()

    pipe = models.logistic_baseline(numeric_cols)
    cv = 3 if quick else 5
    proba = cross_val_predict(pipe, feat, y, cv=cv, method="predict_proba")[:, 1]
    auroc = roc_auc_score(y, proba)

    cox = models.fit_cox(df, ["age", "n_comorbidities", "mean_bp",
                              "serum_creatinine", "cancer"])
    c_index = cox.concordance_index_

    results = {
        "n_patients": int(len(df)),
        "event_rate": float(df["event"].mean()),
        "logistic_auroc_cv": round(float(auroc), 3),
        "cox_c_index": round(float(c_index), 3),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SUPPORT survival pipeline.")
    parser.add_argument("--quick", action="store_true", help="fewer CV folds")
    args = parser.parse_args()

    results = run(quick=args.quick)
    print("Pipeline results")
    print("-" * 32)
    for k, v in results.items():
        print(f"{k:22s}: {v}")


if __name__ == "__main__":
    main()
