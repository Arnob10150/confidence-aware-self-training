from __future__ import annotations

"""Missing-not-at-random (MNAR) label-scarcity simulation (Sec. III-B / Sec. IV-C
"MNAR vs. random hiding").

Standard MCAR hiding (`utils.get_splits`) reveals labels uniformly at random. Real
diagnostic follow-up is not like that: clear-cut, confidently low- or high-risk
patients get definitively diagnosed/confirmed sooner, while ambiguous, mid-risk
patients are disproportionately left for later follow-up (or lost to it) -- so
their labels are disproportionately the ones missing.

We simulate this by: (1) fitting a simple risk-score model (LR) on the full
train partition for each seed, (2) ranking training samples by distance of
predicted probability from 0.5 (|p-0.5|, large = clear-cut, small = ambiguous),
(3) revealing the labeled subset preferentially among the clear-cut end of that
ranking rather than uniformly at random, so ambiguous mid-risk cases are
systematically under-represented in the labeled set relative to MCAR.
"""

import importlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    CONFIGS,
    RESULTS_DIR,
    SEEDS,
    ensure_dirs,
    evaluate,
    load_processed,
    make_models,
    self_train,
)

_ca = importlib.import_module("15_class_aware_crest")
class_aware_self_train = _ca.class_aware_self_train

MNAR_RATIOS = [0.05, 0.40]


def mnar_split(X_train: np.ndarray, y_train: np.ndarray, labeled_ratio: float, seed: int):
    """Reveal labels preferentially for clear-cut (high |risk - 0.5|) cases."""
    scaler = StandardScaler().fit(X_train)
    risk_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    risk_model.fit(scaler.transform(X_train), y_train)
    risk = risk_model.predict_proba(scaler.transform(X_train))[:, 1]
    clearness = np.abs(risk - 0.5)

    rng = np.random.default_rng(seed)
    # Soften a hard rank-cutoff into a sampling-without-replacement scheme weighted
    # toward clear-cut cases, so the labeled set is predominantly but not purely the
    # most clear-cut tail (keeping a mix, as real follow-up would have some noise).
    weights = clearness - clearness.min() + 0.05
    weights = weights / weights.sum()
    n_labeled = max(int(round(labeled_ratio * len(X_train))), 2)
    labeled_idx = rng.choice(len(X_train), size=n_labeled, replace=False, p=weights)
    mask = np.zeros(len(X_train), dtype=bool)
    mask[labeled_idx] = True
    return X_train[mask], y_train[mask], X_train[~mask], y_train[~mask]


def get_splits_scenario(X, y, labeled_ratio: float, seed: int, scenario: str):
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, stratify=y_arr, random_state=seed,
    )
    if scenario == "MCAR":
        X_l, X_u, y_l, y_u = train_test_split(
            X_train, y_train, train_size=labeled_ratio, stratify=y_train, random_state=seed,
        )
    elif scenario == "MNAR":
        X_l, y_l, X_u, y_u = mnar_split(X_train, y_train, labeled_ratio, seed)
    else:
        raise ValueError(scenario)
    return X_l, y_l, X_u, y_u, X_test, y_test


def run_mnar_simulation(ratios=MNAR_RATIOS) -> pd.DataFrame:
    ensure_dirs()
    rows = []
    xgb_spec = make_models()["XGB"]
    for config in CONFIGS:
        X, y = load_processed(config)
        for ratio in ratios:
            for seed in SEEDS:
                for scenario in ["MCAR", "MNAR"]:
                    X_l, y_l, X_u, y_u, X_test, y_test = get_splits_scenario(X, y, ratio, seed, scenario)

                    # Dynamic global-threshold self-training.
                    dyn_model, dyn_scaler, _ = self_train(
                        xgb_spec.estimator_factory, X_l, y_l, X_u, y_u, X_test, y_test,
                        "dynamic", scale=xgb_spec.scale, seed=seed,
                    )
                    rows.append({
                        "config": config, "ratio": ratio, "seed": seed, "scenario": scenario,
                        "method": "DynamicST", **evaluate(dyn_model, X_test, y_test, dyn_scaler),
                    })

                    # Class-aware self-training (reuse the Sec. III-C fix).
                    ca_model, ca_scaler, _ = class_aware_self_train(
                        xgb_spec.estimator_factory, X_l, y_l, X_u, y_u, X_test, y_test,
                        scale=xgb_spec.scale, seed=seed,
                    )
                    rows.append({
                        "config": config, "ratio": ratio, "seed": seed, "scenario": scenario,
                        "method": "ClassAwareST", **evaluate(ca_model, X_test, y_test, ca_scaler),
                    })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "mnar_simulation_raw.csv", index=False)
    summary = out.groupby(["config", "ratio", "scenario", "method"]).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "mnar_simulation_summary.csv", index=False)
    print(summary.to_string(index=False))
    return out


if __name__ == "__main__":
    run_mnar_simulation()
