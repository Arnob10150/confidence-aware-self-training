from __future__ import annotations

"""Class-aware adaptive-threshold self-training (ours) and the CReST baseline.

Both methods are compared against the three global-threshold strategies already
produced by `3_self_training_variants.py` (vanilla/static_high/dynamic) and the
SSL baselines from `4_ssl_comparison.py` (Mean Teacher, Label Propagation).

Class-aware fix (Sec. III-C): following FlexMatch, the global threshold is
scaled down per class in proportion to that class's share of confidently
predicted (but not yet necessarily accepted) samples:

    tau_t(c) = max(tau_min, tau_t * n_c / max_c' n_c')

where tau_t is the same dynamic schedule used elsewhere (0.95 - 0.05*t, floor
0.70), n_c counts unlabeled samples whose current arg-max prediction is class c
(regardless of confidence), and tau_min = 0.55.

CReST (Wei et al. 2021): pseudo-labels are accepted at a single global
threshold (we use the same dynamic schedule for a fair comparison), then the
accepted pool for each class is re-sampled in inverse proportion to that
class's current share of the cumulative labeled pool, so the minority class's
accepted pseudo-labels are up-weighted (via duplication) relative to the
majority class's.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone

from utils import (
    CONFIGS,
    RATIOS,
    RESULTS_DIR,
    SEEDS,
    ensure_dirs,
    evaluate,
    fit_model,
    get_splits,
    load_processed,
    make_models,
    threshold_schedule,
)

TAU_MIN = 0.65
MAX_ITER = 10


def class_aware_self_train(
    model_factory,
    X_l: np.ndarray,
    y_l: np.ndarray,
    X_u: np.ndarray,
    y_u_true: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale: bool,
    seed: int,
    max_iter: int = MAX_ITER,
):
    X_l, y_l = X_l.copy(), y_l.copy()
    X_u, y_u_true = X_u.copy(), y_u_true.copy()
    log: list[dict] = []
    cumulative_added = 0
    cumulative_positive = 0

    for iteration in range(max_iter):
        if len(X_u) == 0:
            break
        model, scaler = fit_model(model_factory(seed), X_l, y_l, scale)
        Xu = scaler.transform(X_u) if scaler else X_u
        proba = model.predict_proba(Xu)
        conf = proba.max(axis=1)
        pseudo = proba.argmax(axis=1)

        tau_t = threshold_schedule("dynamic", iteration)
        # n_c: unlabeled samples currently (arg-max) predicted as class c, regardless of
        # confidence -- FlexMatch's learning-effect proxy for how well class c is learned.
        n_pred = np.array([(pseudo == c).sum() for c in (0, 1)], dtype=float)
        n_max = max(n_pred.max(), 1.0)
        class_ratio = n_pred / n_max
        # Linear interpolation between tau_min and tau_t by class share, floored at
        # tau_min: a minority class with a small predicted share sits near tau_min,
        # a majority class near full share keeps tau_t, with a smooth transition
        # between rather than an immediate drop to the floor.
        per_class_tau = {c: TAU_MIN + (tau_t - TAU_MIN) * class_ratio[c] for c in (0, 1)}
        class_thresholds = np.array([per_class_tau[p] for p in pseudo])
        mask = conf >= class_thresholds

        if int(mask.sum()) == 0:
            break

        added = int(mask.sum())
        positive_added = int(pseudo[mask].sum())
        correct = int((pseudo[mask] == y_u_true[mask]).sum())
        wrong = added - correct
        cumulative_added += added
        cumulative_positive += positive_added

        iter_model, iter_scaler = fit_model(
            model_factory(seed), np.vstack([X_l, X_u[mask]]), np.concatenate([y_l, pseudo[mask]]), scale
        )
        iter_metrics = evaluate(iter_model, X_test, y_test, iter_scaler)
        log.append(
            {
                "iter": iteration,
                "thr_minority": min(per_class_tau.values()),
                "thr_majority": max(per_class_tau.values()),
                "added": added,
                "pl_acc": correct / added,
                "pl_pos_rate": positive_added / added,
                "cumulative_added": cumulative_added,
                "cumulative_positive": cumulative_positive,
                "cumulative_pos_rate": cumulative_positive / cumulative_added,
                "wrong_added": wrong,
                "cumulative_error_rate": wrong / added,
                "test_auc_after_iter": iter_metrics["auc"],
            }
        )

        X_l = np.vstack([X_l, X_u[mask]])
        y_l = np.concatenate([y_l, pseudo[mask]])
        X_u = X_u[~mask]
        y_u_true = y_u_true[~mask]

    final_model, final_scaler = fit_model(model_factory(seed), X_l, y_l, scale)
    return final_model, final_scaler, log


def crest_self_train(
    model_factory,
    X_l: np.ndarray,
    y_l: np.ndarray,
    X_u: np.ndarray,
    y_u_true: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale: bool,
    seed: int,
    rng: np.random.Generator,
    max_iter: int = MAX_ITER,
):
    """CReST: single global (dynamic) threshold to accept, then per-class
    inverse-proportion resampling (duplication) of accepted pseudo-labels
    relative to each class's current share of the labeled pool."""
    X_l, y_l = X_l.copy(), y_l.copy()
    X_u, y_u_true = X_u.copy(), y_u_true.copy()
    log: list[dict] = []

    for iteration in range(max_iter):
        if len(X_u) == 0:
            break
        model, scaler = fit_model(model_factory(seed), X_l, y_l, scale)
        Xu = scaler.transform(X_u) if scaler else X_u
        proba = model.predict_proba(Xu)
        conf = proba.max(axis=1)
        pseudo = proba.argmax(axis=1)
        tau_t = threshold_schedule("dynamic", iteration)
        mask = conf >= tau_t
        if int(mask.sum()) == 0:
            break

        added = int(mask.sum())
        accepted_X = X_u[mask]
        accepted_y = pseudo[mask]
        correct = int((accepted_y == y_u_true[mask]).sum())

        # Current labeled-pool class share -> inverse-proportion resample factor.
        class_counts = np.array([(y_l == c).sum() for c in (0, 1)], dtype=float)
        class_share = class_counts / max(class_counts.sum(), 1.0)
        # CReST's resampling factor alpha=1 (full inverse proportion), clipped to [1, 5]x duplication.
        resample_X, resample_y = [], []
        for c in (0, 1):
            c_mask = accepted_y == c
            if c_mask.sum() == 0:
                continue
            factor = np.clip((1.0 - class_share[c]) / max(class_share[c], 1e-6), 1.0, 1.6)
            n_dup = int(round(factor))
            resample_X.append(np.repeat(accepted_X[c_mask], n_dup, axis=0))
            resample_y.append(np.repeat(accepted_y[c_mask], n_dup, axis=0))
        resampled_X = np.vstack(resample_X) if resample_X else accepted_X
        resampled_y = np.concatenate(resample_y) if resample_y else accepted_y

        iter_model, iter_scaler = fit_model(
            model_factory(seed), np.vstack([X_l, resampled_X]), np.concatenate([y_l, resampled_y]), scale
        )
        iter_metrics = evaluate(iter_model, X_test, y_test, iter_scaler)
        log.append(
            {
                "iter": iteration,
                "thr": tau_t,
                "added": added,
                "pl_acc": correct / added,
                "test_auc_after_iter": iter_metrics["auc"],
            }
        )

        X_l = np.vstack([X_l, resampled_X])
        y_l = np.concatenate([y_l, resampled_y])
        X_u = X_u[~mask]
        y_u_true = y_u_true[~mask]

    final_model, final_scaler = fit_model(model_factory(seed), X_l, y_l, scale)
    return final_model, final_scaler, log


def run_class_aware_crest(ratios=(0.05, 0.40)) -> pd.DataFrame:
    ensure_dirs()
    rows = []
    iter_rows = []
    xgb_spec = make_models()["XGB"]
    for config in CONFIGS:
        X, y = load_processed(config)
        for ratio in ratios:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)

                ca_model, ca_scaler, ca_log = class_aware_self_train(
                    xgb_spec.estimator_factory, X_l, y_l, X_u, y_u, X_test, y_test,
                    scale=xgb_spec.scale, seed=seed,
                )
                rows.append({
                    "config": config, "ratio": ratio, "seed": seed, "method": "ClassAwareST",
                    "iterations": len(ca_log),
                    **evaluate(ca_model, X_test, y_test, ca_scaler),
                })
                for item in ca_log:
                    iter_rows.append({"config": config, "ratio": ratio, "seed": seed, "method": "ClassAwareST", **item})

                crest_model, crest_scaler, crest_log = crest_self_train(
                    xgb_spec.estimator_factory, X_l, y_l, X_u, y_u, X_test, y_test,
                    scale=xgb_spec.scale, seed=seed, rng=rng,
                )
                rows.append({
                    "config": config, "ratio": ratio, "seed": seed, "method": "CReST",
                    "iterations": len(crest_log),
                    **evaluate(crest_model, X_test, y_test, crest_scaler),
                })
                for item in crest_log:
                    iter_rows.append({"config": config, "ratio": ratio, "seed": seed, "method": "CReST", **item})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "class_aware_crest_raw.csv", index=False)
    iters = pd.DataFrame(iter_rows)
    iters.to_csv(RESULTS_DIR / "class_aware_crest_iteration_logs.csv", index=False)
    summary = out.groupby(["config", "ratio", "method"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "class_aware_crest_summary.csv", index=False)
    print(summary.to_string(index=False))
    return out


if __name__ == "__main__":
    run_class_aware_crest()
