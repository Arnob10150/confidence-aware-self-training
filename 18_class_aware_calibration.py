from __future__ import annotations

"""Extends `11_recalibration_dca.py`'s calibration/DCA analysis to Class-Aware ST XGB,
reusing its exact Brier/ECE/Platt/isotonic/net-benefit functions, so Table III's
"Class-Aware ST XGB" row and the decision-curve-analysis paragraph are backed by the
same methodology as the Supervised/Dynamic ST rows already in recalibration_metrics.csv.
"""

import importlib

import numpy as np
import pandas as pd

from utils import RESULTS_DIR, SEEDS, ensure_dirs, get_splits, load_processed, make_models

_base = importlib.import_module("11_recalibration_dca")
_ca = importlib.import_module("15_class_aware_crest")

PRIMARY = "framingham"
CAL_RATIOS = [0.05, 0.40]
DCA_RATIO = 0.40
THRESHOLDS = _base.THRESHOLDS


def collect_class_aware_predictions() -> list[dict]:
    X, y = load_processed(PRIMARY)
    spec = make_models()["XGB"]
    records = []
    for ratio in CAL_RATIOS:
        for seed in SEEDS:
            X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)
            ca_model, ca_scaler, _ = _ca.class_aware_self_train(
                spec.estimator_factory, X_l, y_l, X_u, y_u, X_test, y_test,
                scale=spec.scale, seed=seed,
            )
            Xt = ca_scaler.transform(X_test) if ca_scaler else X_test
            prob = ca_model.predict_proba(Xt)[:, 1]
            records.append({
                "ratio": ratio, "seed": seed, "method": "Class-Aware ST XGB",
                "prob": np.asarray(prob), "true": np.asarray(y_test),
            })
    return records


def run() -> None:
    ensure_dirs()
    records = collect_class_aware_predictions()

    # Calibration/recalibration: append to the existing recalibration_metrics.csv.
    new_rows = _base.run_recalibration_analysis.__wrapped__ if hasattr(_base.run_recalibration_analysis, "__wrapped__") else None
    # run_recalibration_analysis overwrites the CSV, so replicate its row-building logic
    # directly here and merge with the existing file instead of calling it.
    rows = []
    for rec in records:
        prob, true = rec["prob"], rec["true"]
        from sklearn.model_selection import train_test_split
        calib_idx, eval_idx = train_test_split(
            np.arange(len(true)), test_size=0.5, stratify=true, random_state=rec["seed"]
        )
        p_calib, y_calib = prob[calib_idx], true[calib_idx]
        p_eval, y_eval = prob[eval_idx], true[eval_idx]

        raw_brier, raw_ece = _base.brier_score(y_eval, p_eval), _base.expected_calibration_error(y_eval, p_eval)

        from sklearn.linear_model import LogisticRegression
        platt = LogisticRegression()
        platt.fit(p_calib.reshape(-1, 1), y_calib)
        p_platt = platt.predict_proba(p_eval.reshape(-1, 1))[:, 1]
        platt_brier, platt_ece = _base.brier_score(y_eval, p_platt), _base.expected_calibration_error(y_eval, p_platt)

        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_calib, y_calib)
        p_iso = iso.predict(p_eval)
        iso_brier, iso_ece = _base.brier_score(y_eval, p_iso), _base.expected_calibration_error(y_eval, p_iso)

        rows.append({
            "ratio": rec["ratio"], "seed": rec["seed"], "method": rec["method"],
            "raw_brier": raw_brier, "raw_ece": raw_ece,
            "platt_brier": platt_brier, "platt_ece": platt_ece,
            "isotonic_brier": iso_brier, "isotonic_ece": iso_ece,
        })
    new_df = pd.DataFrame(rows)

    existing_path = RESULTS_DIR / "recalibration_metrics.csv"
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    existing = existing[existing["method"] != "Class-Aware ST XGB"]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(existing_path, index=False)
    summary = combined.groupby(["ratio", "method"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    summary.to_csv(RESULTS_DIR / "recalibration_summary.csv", index=False)
    print("Updated recalibration summary:")
    print(summary.to_string(index=False))

    # Decision-curve analysis at 40% labels: append to decision_curve_raw.csv.
    dca_rows = []
    for rec in records:
        if rec["ratio"] != DCA_RATIO:
            continue
        prob, true = rec["prob"], rec["true"]
        prevalence = true.mean()
        for thr in THRESHOLDS:
            dca_rows.append({
                "method": rec["method"], "seed": rec["seed"], "threshold": thr,
                "net_benefit_model": _base.net_benefit(prob, true, thr),
                "net_benefit_treat_all": prevalence - (1 - prevalence) * (thr / (1 - thr)),
            })
    new_dca = pd.DataFrame(dca_rows)
    dca_path = RESULTS_DIR / "decision_curve_raw.csv"
    existing_dca = pd.read_csv(dca_path) if dca_path.exists() else pd.DataFrame()
    existing_dca = existing_dca[existing_dca["method"] != "Class-Aware ST XGB"]
    combined_dca = pd.concat([existing_dca, new_dca], ignore_index=True)
    combined_dca.to_csv(dca_path, index=False)
    dca_summary = combined_dca.groupby(["method", "threshold"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    dca_summary.to_csv(RESULTS_DIR / "decision_curve_summary.csv", index=False)
    print("\nUpdated decision-curve net benefit at 40% labels:")
    print(dca_summary.to_string(index=False))


if __name__ == "__main__":
    run()
