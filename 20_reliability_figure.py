from __future__ import annotations

"""Regenerates figures/reliability_diagram.pdf with all three curves the paper
actually discusses (Supervised XGB, Dynamic Self-Training XGB, Class-Aware ST
XGB + Platt), using real per-sample predicted probabilities collected from the
same regenerated pipelines as class_aware_crest_raw.csv / recalibration_metrics.csv.

The pre-existing figures/reliability_diagram.pdf (from lost original code) only
had two lines when regenerated via 9_calibration_analysis.py; this script adds
the third curve with real data and uses Type 42 (TrueType) fonts for IEEE PDF
eXpress compliance, matching 9_calibration_analysis.py's visual style.
"""

import importlib

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import FIGURES_DIR, SEEDS, get_splits, load_processed, make_models, self_train

_ca = importlib.import_module("15_class_aware_crest")

PRIMARY = "framingham"
CAL_RATIOS = [0.05, 0.40]
N_BINS = 10


def collect_curves() -> pd.DataFrame:
    X, y = load_processed(PRIMARY)
    spec = make_models()["XGB"]
    curve_rows = []
    for ratio in CAL_RATIOS:
        for seed in SEEDS:
            X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)

            from utils import fit_model
            sup_model, sup_scaler = fit_model(spec.estimator_factory(seed), X_l, y_l, spec.scale)
            Xt_sup = sup_scaler.transform(X_test) if sup_scaler else X_test
            sup_prob = sup_model.predict_proba(Xt_sup)[:, 1]

            st_model, st_scaler, _ = self_train(
                spec.estimator_factory, X_l, y_l, X_u, y_u, X_test, y_test,
                "dynamic", scale=spec.scale, seed=seed,
            )
            Xt_st = st_scaler.transform(X_test) if st_scaler else X_test
            st_prob = st_model.predict_proba(Xt_st)[:, 1]

            ca_model, ca_scaler, _ = _ca.class_aware_self_train(
                spec.estimator_factory, X_l, y_l, X_u, y_u, X_test, y_test,
                scale=spec.scale, seed=seed,
            )
            Xt_ca = ca_scaler.transform(X_test) if ca_scaler else X_test
            ca_prob_raw = ca_model.predict_proba(Xt_ca)[:, 1]
            # Default-pipeline Platt recalibration: split test into calib/eval halves,
            # matching Sec. III-D's methodology exactly.
            calib_idx, eval_idx = train_test_split(
                np.arange(len(y_test)), test_size=0.5, stratify=y_test, random_state=seed
            )
            platt = LogisticRegression()
            platt.fit(ca_prob_raw[calib_idx].reshape(-1, 1), np.asarray(y_test)[calib_idx])
            ca_prob_platt = platt.predict_proba(ca_prob_raw[eval_idx].reshape(-1, 1))[:, 1]
            y_eval = np.asarray(y_test)[eval_idx]

            for p, t in zip(sup_prob, y_test):
                curve_rows.append({"ratio": ratio, "method": "Supervised XGB", "prob": float(p), "true": int(t)})
            for p, t in zip(st_prob, y_test):
                curve_rows.append({"ratio": ratio, "method": "Dynamic Self-Training XGB", "prob": float(p), "true": int(t)})
            for p, t in zip(ca_prob_platt, y_eval):
                curve_rows.append({"ratio": ratio, "method": "Class-Aware ST XGB + Platt", "prob": float(p), "true": int(t)})
    return pd.DataFrame(curve_rows)


def plot_reliability(curves: pd.DataFrame) -> None:
    bins = np.linspace(0, 1, N_BINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 6.4))
    styles = {
        "Supervised XGB": ("#D85A30", "--", "^"),
        "Dynamic Self-Training XGB": ("#185FA5", "-", "o"),
        "Class-Aware ST XGB + Platt": ("#1D9E75", "-", "s"),
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
    print("Saved results/figures/reliability_diagram.pdf")


if __name__ == "__main__":
    curves = collect_curves()
    plot_reliability(curves)
