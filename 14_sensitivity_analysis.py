from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from utils import RESULTS_DIR, SEEDS, ensure_dirs, evaluate, fit_model, get_splits, load_processed

PRIMARY = "framingham"
POS_CLASS_WEIGHT = 3596 / 644
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
HYPERPARAM_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in (80, 120, 160)
    for d in (2, 3, 4)
    for lr in (0.03, 0.05, 0.08)
]


def make_xgb(seed: int, **overrides):
    params = dict(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=POS_CLASS_WEIGHT,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )
    params.update(overrides)
    return XGBClassifier(**params)


def self_train_fixed_threshold(X_l, y_l, X_u, y_u_true, X_test, y_test, tau: float, seed: int, max_iter: int = 10):
    X_l, y_l, X_u, y_u_true = X_l.copy(), y_l.copy(), X_u.copy(), y_u_true.copy()
    log: list[dict] = []
    for _ in range(max_iter):
        if len(X_u) == 0:
            break
        model, scaler = fit_model(make_xgb(seed), X_l, y_l, scale=False)
        proba = model.predict_proba(X_u)
        conf = proba.max(axis=1)
        pseudo = proba.argmax(axis=1)
        mask = conf >= tau
        if int(mask.sum()) == 0:
            break
        added = int(mask.sum())
        positive_added = int(pseudo[mask].sum())
        log.append({"added": added, "pl_pos_rate": positive_added / added})
        X_l = np.vstack([X_l, X_u[mask]])
        y_l = np.concatenate([y_l, pseudo[mask]])
        X_u = X_u[~mask]
        y_u_true = y_u_true[~mask]
    final_model, final_scaler = fit_model(make_xgb(seed), X_l, y_l, scale=False)
    return final_model, final_scaler, log


def threshold_sensitivity(ratio: float = 0.40) -> pd.DataFrame:
    X, y = load_processed(PRIMARY)
    rows = []
    for tau in THRESHOLDS:
        for seed in SEEDS:
            X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)
            model, scaler, log = self_train_fixed_threshold(X_l, y_l, X_u, y_u, X_test, y_test, tau, seed)
            metrics = evaluate(model, X_test, y_test, scaler)
            pos_rate = np.mean([item["pl_pos_rate"] for item in log]) if log else float("nan")
            rows.append({"tau": tau, "seed": seed, "auc": metrics["auc"], "pl_pos_rate": pos_rate})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "threshold_sensitivity_raw.csv", index=False)
    summary = df.groupby("tau").mean(numeric_only=True).drop(columns="seed").reset_index()
    summary.to_csv(RESULTS_DIR / "threshold_sensitivity_summary.csv", index=False)
    return summary


def hyperparameter_sensitivity(seed: int = 42) -> pd.DataFrame:
    X, y = load_processed(PRIMARY)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for params in HYPERPARAM_GRID:
        aucs = []
        for train_idx, test_idx in skf.split(X, y):
            model, scaler = fit_model(make_xgb(seed, **params), X.iloc[train_idx].values, y.iloc[train_idx].values, scale=False)
            metrics = evaluate(model, X.iloc[test_idx].values, y.iloc[test_idx].values, scaler)
            aucs.append(metrics["auc"])
        rows.append({**params, "cv_auc_mean": np.mean(aucs), "cv_auc_std": np.std(aucs)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "hyperparam_sensitivity.csv", index=False)
    return df


def run() -> None:
    ensure_dirs()
    thr = threshold_sensitivity()
    print("Threshold sensitivity (40% labels, XGB):")
    print(thr.to_string(index=False))

    hp = hyperparameter_sensitivity()
    print(f"\nHyperparameter grid ({len(HYPERPARAM_GRID)} combos), full-label 5-fold CV AUC:")
    print(f"  min={hp['cv_auc_mean'].min():.4f}  max={hp['cv_auc_mean'].max():.4f}  "
          f"mean={hp['cv_auc_mean'].mean():.4f}  std={hp['cv_auc_mean'].std():.4f}")
    chosen = hp[(hp["n_estimators"] == 120) & (hp["max_depth"] == 3) & (hp["learning_rate"] == 0.05)]
    print("Chosen config (n=120, depth=3, lr=0.05):")
    print(chosen.to_string(index=False))


if __name__ == "__main__":
    run()
