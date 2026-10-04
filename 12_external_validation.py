from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from utils import RESULTS_DIR, ensure_dirs, fit_model, load_processed, make_models

PRIMARY = "framingham"
SHARED_FEATURES = ["age", "male", "sysBP", "totChol", "diabetes"]
EXTERNAL_PATH = "cross_check/heart.csv"


def load_external() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_PATH)
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    df["Cholesterol"] = df["Cholesterol"].fillna(df["Cholesterol"].median())
    df["RestingBP"] = df["RestingBP"].fillna(df["RestingBP"].median())
    out = pd.DataFrame(
        {
            "age": df["Age"],
            "male": (df["Sex"] == "M").astype(int),
            "sysBP": df["RestingBP"],
            "totChol": df["Cholesterol"],
            "diabetes": df["FastingBS"],
            "HeartDisease": df["HeartDisease"],
        }
    )
    return out


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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


def reduced_feature_cv_ceiling(X: pd.DataFrame, y: pd.Series, model_name: str, seed: int = 42) -> dict:
    spec = make_models()[model_name]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for train_idx, test_idx in skf.split(X, y):
        model, scaler = fit_model(spec.estimator_factory(seed), X.iloc[train_idx].values, y.iloc[train_idx].values, spec.scale)
        Xt = scaler.transform(X.iloc[test_idx].values) if scaler else X.iloc[test_idx].values
        prob = model.predict_proba(Xt)[:, 1]
        pred = model.predict(Xt)
        rows.append(
            {
                "auc": roc_auc_score(y.iloc[test_idx], prob),
                "f1": f1_score(y.iloc[test_idx], pred, zero_division=0),
                "recall": recall_score(y.iloc[test_idx], pred, zero_division=0),
            }
        )
    df = pd.DataFrame(rows)
    return {"cv_auc_mean": df["auc"].mean(), "cv_f1_mean": df["f1"].mean(), "cv_recall_mean": df["recall"].mean()}


def run(seed: int = 42) -> pd.DataFrame:
    ensure_dirs()
    X, y = load_processed(PRIMARY)
    X_shared = X[SHARED_FEATURES]
    external = load_external()
    X_ext, y_ext = external[SHARED_FEATURES], external["HeartDisease"]

    rows = []
    for model_name in ["LR", "XGB"]:
        spec = make_models()[model_name]
        ceiling = reduced_feature_cv_ceiling(X_shared, y, model_name, seed)

        model, scaler = fit_model(spec.estimator_factory(seed), X_shared.values, y.values, spec.scale)
        Xt_ext = scaler.transform(X_ext.values) if scaler else X_ext.values
        prob_ext = model.predict_proba(Xt_ext)[:, 1]
        pred_ext = model.predict(Xt_ext)

        rows.append(
            {
                "model": model_name,
                "framingham_cv_auc": ceiling["cv_auc_mean"],
                "framingham_cv_f1": ceiling["cv_f1_mean"],
                "framingham_cv_recall": ceiling["cv_recall_mean"],
                "external_auc": roc_auc_score(y_ext, prob_ext),
                "external_f1": f1_score(y_ext, pred_ext, zero_division=0),
                "external_recall": recall_score(y_ext, pred_ext, zero_division=0),
                "external_precision": precision_score(y_ext, pred_ext, zero_division=0),
                "external_accuracy": accuracy_score(y_ext, pred_ext),
                "external_brier": brier_score(y_ext.values, prob_ext),
                "external_ece": expected_calibration_error(y_ext.values, prob_ext),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "external_validation.csv", index=False)
    print(f"External cohort: n={len(y_ext)}, prevalence={y_ext.mean():.3f}")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
