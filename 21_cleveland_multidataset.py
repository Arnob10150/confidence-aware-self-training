from __future__ import annotations

"""Real UCI Cleveland (303 patients, Detrano et al. 1989) generalizability analysis
and regeneration of figures/multidataset_combined.pdf.

Replaces the previous figure, whose numbers had no generating script in this repo
and whose only Cleveland-like data file (cross_check/heart.csv) was in fact the
918-row combined UCI heart-disease cohort (Cleveland+Hungary+Switzerland+VA), not
the canonical 303-row Cleveland set the paper describes and cites.

Data: cross_check/uci_cleveland_processed.csv, derived from UCI ML Repository
processed.cleveland.data (303 rows, 13 predictors, target binarized 0 vs 1-4,
giving 139 positives = 45.9% prevalence, matching the paper's Sec. III-A text).

Produces:
  results/cleveland_cv_ceiling.csv   full-label 5-fold CV per model, both cohorts
  results/cleveland_selftraining.csv dynamic XGB self-training AUC by labeled ratio
  results/figures/multidataset_combined.pdf
"""

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # Type 42 (TrueType), not Type 3, for IEEE PDF eXpress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import (
    FIGURES_DIR,
    RATIOS,
    RESULTS_DIR,
    SEEDS,
    ensure_dirs,
    evaluate,
    fit_model,
    get_splits,
    load_processed,
    make_models,
    self_train,
)

CLEVELAND_PATH = "cross_check/uci_cleveland_processed.csv"
MODELS = ["LR", "RF", "SVM", "XGB"]


def load_cleveland() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(CLEVELAND_PATH)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    return X, y


def cv_ceiling(X: pd.DataFrame, y: pd.Series, model_name: str, seed: int = 42) -> float:
    """Full-label 5-fold CV AUC, same protocol as utils.supervised_cv_ceiling."""
    spec = make_models()[model_name]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        model, scaler = fit_model(
            spec.estimator_factory(seed), X.iloc[train_idx].values, y.iloc[train_idx].values, spec.scale
        )
        aucs.append(evaluate(model, X.iloc[test_idx].values, y.iloc[test_idx].values, scaler)["auc"])
    return float(np.mean(aucs))


def run() -> None:
    ensure_dirs()
    X_fram, y_fram = load_processed("framingham")
    X_clev, y_clev = load_cleveland()
    print(f"Cleveland: n={len(y_clev)}, positives={int(y_clev.sum())} ({y_clev.mean():.3f})")

    # --- Panel (a): full-label CV ceiling per model, both cohorts ---
    ceiling_rows = []
    for model_name in MODELS:
        ceiling_rows.append({
            "model": model_name,
            "framingham_cv_auc": cv_ceiling(X_fram, y_fram, model_name),
            "cleveland_cv_auc": cv_ceiling(X_clev, y_clev, model_name),
        })
    ceiling = pd.DataFrame(ceiling_rows)
    ceiling.to_csv(RESULTS_DIR / "cleveland_cv_ceiling.csv", index=False)
    print(ceiling.to_string(index=False))

    # --- Panel (b): dynamic XGB self-training AUC by labeled ratio, both cohorts ---
    xgb = make_models()["XGB"]
    st_rows = []
    for cohort, (X, y) in [("Framingham", (X_fram, y_fram)), ("UCI Cleveland", (X_clev, y_clev))]:
        for ratio in RATIOS:
            for seed in SEEDS:
                X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)
                if len(np.unique(y_l)) < 2:
                    continue  # too few labels at this ratio for this small cohort
                model, scaler, _ = self_train(
                    xgb.estimator_factory, X_l, y_l, X_u, y_u, X_test, y_test,
                    "dynamic", scale=xgb.scale, seed=seed,
                )
                st_rows.append({
                    "cohort": cohort, "ratio": ratio, "seed": seed,
                    **evaluate(model, X_test, y_test, scaler),
                })
    st = pd.DataFrame(st_rows)
    st.to_csv(RESULTS_DIR / "cleveland_selftraining.csv", index=False)
    print(st.groupby(["cohort", "ratio"])["auc"].mean().to_string())

    # --- Figure ---
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.8))

    width = 0.38
    xpos = np.arange(len(MODELS))
    axes[0].bar(xpos - width / 2, ceiling["cleveland_cv_auc"], width, label="UCI Cleveland", color="#185FA5")
    axes[0].bar(xpos + width / 2, ceiling["framingham_cv_auc"], width, label="Framingham", color="#D85A30")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(MODELS)
    axes[0].set_xlabel("Model", fontsize=11)
    axes[0].set_ylabel("Full-label CV AUC", fontsize=11)
    axes[0].set_title("(a) Ceiling AUC by dataset", fontsize=11)
    axes[0].tick_params(labelsize=10)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=9, frameon=False)
    axes[0].grid(alpha=0.12, axis="y")

    styles = {"Framingham": ("#185FA5", "-", "o"), "UCI Cleveland": ("#D85A30", "--", "^")}
    for cohort, (color, ls, marker) in styles.items():
        g = st[st["cohort"] == cohort].groupby("ratio")["auc"].mean()
        axes[1].plot(g.index.to_numpy(), g.to_numpy(), color=color, linestyle=ls, marker=marker,
                     markersize=4.5, linewidth=1.8, label=cohort)
    axes[1].set_xlabel("Labeled training ratio", fontsize=11)
    axes[1].set_ylabel("Dynamic ST AUC (XGB)", fontsize=11)
    axes[1].set_title("(b) Self-training AUC by dataset", fontsize=11)
    axes[1].tick_params(labelsize=10)
    axes[1].margins(y=0.1)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=9, frameon=False)
    axes[1].grid(alpha=0.12)

    fig.subplots_adjust(hspace=0.95)
    plt.savefig(FIGURES_DIR / "multidataset_combined.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close()
    print("Saved results/figures/multidataset_combined.pdf")


if __name__ == "__main__":
    run()
