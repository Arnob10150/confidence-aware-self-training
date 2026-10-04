from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation

from utils import CONFIGS, RATIOS, SEEDS, ensure_dirs, evaluate, get_splits, load_processed, make_models, self_train


def metric_dict(y_true, pred, prob) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob),
    }


def label_propagation(X_l, y_l, X_u, X_test, y_test) -> dict[str, float]:
    X_combined = np.vstack([X_l, X_u])
    y_combined = np.concatenate([y_l, -np.ones(len(X_u), dtype=int)])
    scaler = StandardScaler().fit(X_combined)
    model = LabelPropagation(kernel="rbf", gamma=20, max_iter=1000)
    model.fit(scaler.transform(X_combined), y_combined)
    Xt = scaler.transform(X_test)
    prob = model.predict_proba(Xt)[:, 1]
    prob = np.nan_to_num(prob, nan=float(np.mean(y_l)), posinf=1.0, neginf=0.0)
    pred = (prob >= 0.5).astype(int)
    return metric_dict(y_test, pred, prob)


def mean_teacher(X_l, y_l, X_u, X_test, y_test, seed=42, ema=0.99, rounds=10) -> dict[str, float]:
    base = make_models()["XGB"].estimator_factory(seed)
    scaler = StandardScaler().fit(np.vstack([X_l, X_u]))
    X_l_s = scaler.transform(X_l)
    X_u_s = scaler.transform(X_u)
    X_test_s = scaler.transform(X_test)
    teacher_prob = None
    X_train = X_l_s.copy()
    y_train = y_l.copy()
    for _round in range(rounds):
        student = clone(base).fit(X_train, y_train)
        student_prob = student.predict_proba(X_u_s)[:, 1]
        teacher_prob = student_prob if teacher_prob is None else ema * teacher_prob + (1 - ema) * student_prob
        confidence = np.abs(teacher_prob - 0.5) * 2
        mask = confidence >= 0.8
        if int(mask.sum()) == 0:
            break
        X_train = np.vstack([X_l_s, X_u_s[mask]])
        y_train = np.concatenate([y_l, (teacher_prob[mask] > 0.5).astype(int)])
    final = clone(base).fit(X_train, y_train)
    pred = final.predict(X_test_s)
    prob = final.predict_proba(X_test_s)[:, 1]
    return metric_dict(y_test, pred, prob)


def run_ssl_comparison() -> pd.DataFrame:
    ensure_dirs()
    rows = []
    xgb_spec = make_models()["XGB"]
    for config in CONFIGS:
        X, y = load_processed(config)
        for ratio in RATIOS:
            for seed in SEEDS:
                X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)
                st_model, st_scaler, _log = self_train(
                    xgb_spec.estimator_factory,
                    X_l,
                    y_l,
                    X_u,
                    y_u,
                    X_test,
                    y_test,
                    "dynamic",
                    scale=xgb_spec.scale,
                    seed=seed,
                )
                rows.append(
                    {
                        "config": config,
                        "ratio": ratio,
                        "seed": seed,
                        "method": "SelfTrainingDynamic",
                        **evaluate(st_model, X_test, y_test, st_scaler),
                    }
                )
                rows.append(
                    {
                        "config": config,
                        "ratio": ratio,
                        "seed": seed,
                        "method": "MeanTeacher",
                        **mean_teacher(X_l, y_l, X_u, X_test, y_test, seed=seed),
                    }
                )
                rows.append(
                    {
                        "config": config,
                        "ratio": ratio,
                        "seed": seed,
                        "method": "LabelPropagation",
                        **label_propagation(X_l, y_l, X_u, X_test, y_test),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv("results/ssl_comparison.csv", index=False)
    print("Saved results/ssl_comparison.csv", out.shape)
    return out


if __name__ == "__main__":
    run_ssl_comparison()
