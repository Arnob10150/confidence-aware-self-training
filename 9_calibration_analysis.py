from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # Type 42 (TrueType), not Type 3, for IEEE PDF eXpress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    SEEDS,
    ensure_dirs,
    fit_model,
    get_splits,
    load_processed,
    make_models,
    self_train,
)

PRIMARY = "framingham"
CAL_RATIOS = [0.05, 0.40]
N_BINS = 10


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
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def collect_probabilities() -> tuple[pd.DataFrame, pd.DataFrame]:
    X, y = load_processed(PRIMARY)
    spec = make_models()["XGB"]
    rows = []
    curve_rows = []
    for ratio in CAL_RATIOS:
        for seed in SEEDS:
            X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)

            sup_model, sup_scaler = fit_model(spec.estimator_factory(seed), X_l, y_l, spec.scale)
            Xt_sup = sup_scaler.transform(X_test) if sup_scaler else X_test
            sup_prob = sup_model.predict_proba(Xt_sup)[:, 1]

            st_model, st_scaler, _ = self_train(
                spec.estimator_factory,
                X_l,
                y_l,
                X_u,
                y_u,
                X_test,
                y_test,
                "dynamic",
                scale=spec.scale,
                seed=seed,
            )
            Xt_st = st_scaler.transform(X_test) if st_scaler else X_test
            st_prob = st_model.predict_proba(Xt_st)[:, 1]

            for method, prob in [("Supervised XGB", sup_prob), ("Dynamic Self-Training XGB", st_prob)]:
                prob = np.asarray(prob)
                y_arr = np.asarray(y_test)
                rows.append(
                    {
                        "ratio": ratio,
                        "seed": seed,
                        "method": method,
                        "brier": brier_score(y_arr, prob),
                        "ece": expected_calibration_error(y_arr, prob),
                    }
                )
                for p, t in zip(prob, y_arr):
                    curve_rows.append({"ratio": ratio, "method": method, "prob": float(p), "true": int(t)})
    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics.groupby(["ratio", "method"])[["brier", "ece"]].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


def plot_reliability(curves: pd.DataFrame) -> None:
    bins = np.linspace(0, 1, N_BINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 6.4))
    styles = {
        "Supervised XGB": ("#D85A30", "--", "^"),
        "Dynamic Self-Training XGB": ("#185FA5", "-", "o"),
    }
    for ax, ratio in zip(axes, CAL_RATIOS):
        sub = curves[curves["ratio"] == ratio]
        for method, (color, ls, marker) in styles.items():
            m = sub[sub["method"] == method]
            observed = []
            for i in range(N_BINS):
                lo, hi = bins[i], bins[i + 1]
                mask = (m["prob"] >= lo) & (m["prob"] <= hi if i == N_BINS - 1 else m["prob"] < hi)
                observed.append(m.loc[mask, "true"].mean() if mask.sum() > 0 else np.nan)
            ax.plot(centers, observed, color=color, linestyle=ls, marker=marker, markersize=5, linewidth=1.8, label=method)
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{int(ratio * 100)}% labels", fontsize=11)
        ax.set_xlabel("Mean predicted probability", fontsize=11)
        ax.set_ylabel("Observed positive frequency", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.15)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=9, frameon=False)
    fig.subplots_adjust(hspace=0.95)
    plt.savefig(FIGURES_DIR / "reliability_diagram.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close()


def run_calibration_analysis() -> pd.DataFrame:
    ensure_dirs()
    metrics, curves = collect_probabilities()
    metrics.to_csv(RESULTS_DIR / "calibration_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(RESULTS_DIR / "calibration_summary.csv", index=False)
    plot_reliability(curves)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_calibration_analysis()
