from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import RESULTS_DIR, SEEDS, ensure_dirs, fit_model, get_splits, load_processed, make_models, self_train

PRIMARY = "framingham"
CAL_RATIOS = [0.05, 0.40]
DCA_RATIO = 0.40
N_BINS = 10
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def collect_test_predictions() -> list[dict]:
    X, y = load_processed(PRIMARY)
    spec = make_models()["XGB"]
    records = []
    for ratio in CAL_RATIOS:
        for seed in SEEDS:
            X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)

            sup_model, sup_scaler = fit_model(spec.estimator_factory(seed), X_l, y_l, spec.scale)
            Xt_sup = sup_scaler.transform(X_test) if sup_scaler else X_test
            sup_prob = sup_model.predict_proba(Xt_sup)[:, 1]

            st_model, st_scaler, _ = self_train(
                spec.estimator_factory, X_l, y_l, X_u, y_u, X_test, y_test, "dynamic", scale=spec.scale, seed=seed
            )
            Xt_st = st_scaler.transform(X_test) if st_scaler else X_test
            st_prob = st_model.predict_proba(Xt_st)[:, 1]

            for method, prob in [("Supervised XGB", sup_prob), ("Dynamic Self-Training XGB", st_prob)]:
                records.append(
                    {
                        "ratio": ratio,
                        "seed": seed,
                        "method": method,
                        "prob": np.asarray(prob),
                        "true": np.asarray(y_test),
                    }
                )
    return records


def run_recalibration_analysis(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        prob, true = rec["prob"], rec["true"]
        calib_idx, eval_idx = train_test_split(
            np.arange(len(true)), test_size=0.5, stratify=true, random_state=rec["seed"]
        )
        p_calib, y_calib = prob[calib_idx], true[calib_idx]
        p_eval, y_eval = prob[eval_idx], true[eval_idx]

        raw_brier, raw_ece = brier_score(y_eval, p_eval), expected_calibration_error(y_eval, p_eval)

        platt = LogisticRegression()
        platt.fit(p_calib.reshape(-1, 1), y_calib)
        p_platt = platt.predict_proba(p_eval.reshape(-1, 1))[:, 1]
        platt_brier, platt_ece = brier_score(y_eval, p_platt), expected_calibration_error(y_eval, p_platt)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_calib, y_calib)
        p_iso = iso.predict(p_eval)
        iso_brier, iso_ece = brier_score(y_eval, p_iso), expected_calibration_error(y_eval, p_iso)

        rows.append(
            {
                "ratio": rec["ratio"],
                "seed": rec["seed"],
                "method": rec["method"],
                "raw_brier": raw_brier,
                "raw_ece": raw_ece,
                "platt_brier": platt_brier,
                "platt_ece": platt_ece,
                "isotonic_brier": iso_brier,
                "isotonic_ece": iso_ece,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "recalibration_metrics.csv", index=False)
    summary = df.groupby(["ratio", "method"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    summary.to_csv(RESULTS_DIR / "recalibration_summary.csv", index=False)
    return summary


def net_benefit(prob: np.ndarray, true: np.ndarray, threshold: float) -> float:
    pred_pos = prob >= threshold
    n = len(true)
    tp = int(np.sum(pred_pos & (true == 1)))
    fp = int(np.sum(pred_pos & (true == 0)))
    return tp / n - fp / n * (threshold / (1 - threshold))


def run_decision_curve_analysis(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec["ratio"] != DCA_RATIO:
            continue
        prob, true = rec["prob"], rec["true"]
        prevalence = true.mean()
        for thr in THRESHOLDS:
            rows.append(
                {
                    "method": rec["method"],
                    "seed": rec["seed"],
                    "threshold": thr,
                    "net_benefit_model": net_benefit(prob, true, thr),
                    "net_benefit_treat_all": prevalence - (1 - prevalence) * (thr / (1 - thr)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "decision_curve_raw.csv", index=False)
    summary = df.groupby(["method", "threshold"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    summary.to_csv(RESULTS_DIR / "decision_curve_summary.csv", index=False)
    return summary


def run() -> None:
    ensure_dirs()
    records = collect_test_predictions()
    recal_summary = run_recalibration_analysis(records)
    print("Recalibration summary (5% / 40% labels):")
    print(recal_summary.to_string(index=False))
    dca_summary = run_decision_curve_analysis(records)
    print("\nDecision-curve net benefit at 40% labels:")
    print(dca_summary.to_string(index=False))


if __name__ == "__main__":
    run()
