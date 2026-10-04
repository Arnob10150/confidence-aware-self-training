from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    HAVE_XGBOOST = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier

    HAVE_XGBOOST = False


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FRAMINGHAM_DIR = ROOT / "Framingham"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SHAP_DIR = RESULTS_DIR / "shap"
SEEDS = [42, 1, 2, 3, 4]
RATIOS = [0.01, 0.05, 0.10, 0.20, 0.40]
VARIANTS = ["vanilla", "static_high", "dynamic"]
TARGET_COLUMN = "TenYearCHD"
CONFIGS = ["framingham"]
POS_CLASS_WEIGHT = 3596 / 644


@dataclass
class ModelSpec:
    name: str
    estimator_factory: Callable[[int], object]
    scale: bool


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)


def processed_path(config: str) -> Path:
    if config == "framingham":
        return DATA_DIR / "processed.csv"
    raise ValueError(f"Unknown config: {config}")


def load_processed(config: str = "framingham") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(processed_path(config))
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def get_splits(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    labeled_ratio: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr,
        y_arr,
        test_size=0.2,
        stratify=y_arr,
        random_state=seed,
    )
    if labeled_ratio >= 1.0:
        return (
            X_train,
            y_train,
            np.empty((0, X_arr.shape[1])),
            np.empty(0, dtype=int),
            X_test,
            y_test,
        )
    X_l, X_u, y_l, y_u = train_test_split(
        X_train,
        y_train,
        train_size=labeled_ratio,
        stratify=y_train,
        random_state=seed,
    )
    return X_l, y_l, X_u, y_u, X_test, y_test


def threshold_schedule(variant: str, iteration: int) -> float:
    if variant == "vanilla":
        return 0.80
    if variant == "static_high":
        return 0.95
    if variant == "dynamic":
        return max(0.95 - 0.05 * iteration, 0.70)
    raise ValueError(f"Unknown self-training variant: {variant}")


def make_models() -> dict[str, ModelSpec]:
    def make_xgb(seed: int):
        if HAVE_XGBOOST:
            return XGBClassifier(
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
        return GradientBoostingClassifier(random_state=seed)

    return {
        "LR": ModelSpec(
            "LR",
            lambda seed: LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            ),
            True,
        ),
        "SVM": ModelSpec(
            "SVM",
            lambda seed: SVC(probability=True, class_weight="balanced", random_state=seed),
            True,
        ),
        "RF": ModelSpec(
            "RF",
            lambda seed: RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        ),
        "XGB": ModelSpec("XGB", make_xgb, False),
    }


def fit_model(model, X_train: np.ndarray, y_train: np.ndarray, scale: bool):
    scaler = StandardScaler().fit(X_train) if scale else None
    Xt = scaler.transform(X_train) if scaler else X_train
    fitted = clone(model).fit(Xt, y_train)
    return fitted, scaler


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, scaler=None) -> dict[str, float]:
    Xt = scaler.transform(X_test) if scaler else X_test
    pred = model.predict(Xt)
    prob = model.predict_proba(Xt)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, prob),
    }


def self_train(
    model_factory: Callable[[int], object],
    X_l: np.ndarray,
    y_l: np.ndarray,
    X_u: np.ndarray,
    y_u_true: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    variant: str,
    scale: bool = False,
    seed: int = 42,
    max_iter: int = 10,
) -> tuple[object, object | None, list[dict]]:
    X_l = X_l.copy()
    y_l = y_l.copy()
    X_u = X_u.copy()
    y_u_true = y_u_true.copy()
    log: list[dict] = []
    cumulative_wrong = 0
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
        threshold = threshold_schedule(variant, iteration)
        mask = conf >= threshold
        if int(mask.sum()) == 0:
            break

        added = int(mask.sum())
        positive_added = int(pseudo[mask].sum())
        correct = int((pseudo[mask] == y_u_true[mask]).sum())
        wrong = added - correct
        cumulative_added += added
        cumulative_positive += positive_added
        cumulative_wrong += wrong
        iter_model, iter_scaler = fit_model(
            model_factory(seed), np.vstack([X_l, X_u[mask]]), np.concatenate([y_l, pseudo[mask]]), scale
        )
        iter_metrics = evaluate(iter_model, X_test, y_test, iter_scaler)
        log.append(
            {
                "iter": iteration,
                "thr": threshold,
                "added": added,
                "pl_acc": correct / added,
                "pl_pos_rate": positive_added / added,
                "positive_added": positive_added,
                "negative_added": added - positive_added,
                "noise_rate": wrong / added,
                "wrong_added": wrong,
                "cumulative_added": cumulative_added,
                "cumulative_positive": cumulative_positive,
                "cumulative_pos_rate": cumulative_positive / cumulative_added,
                "cumulative_wrong": cumulative_wrong,
                "cumulative_error_rate": cumulative_wrong / cumulative_added,
                "test_accuracy_after_iter": iter_metrics["accuracy"],
                "test_auc_after_iter": iter_metrics["auc"],
            }
        )

        X_l = np.vstack([X_l, X_u[mask]])
        y_l = np.concatenate([y_l, pseudo[mask]])
        X_u = X_u[~mask]
        y_u_true = y_u_true[~mask]

    final_model, final_scaler = fit_model(model_factory(seed), X_l, y_l, scale)
    return final_model, final_scaler, log


def supervised_cv_ceiling(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> dict[str, float]:
    spec = make_models()["XGB"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for train_idx, test_idx in skf.split(X, y):
        model, scaler = fit_model(
            spec.estimator_factory(seed),
            X.iloc[train_idx].values,
            y.iloc[train_idx].values,
            spec.scale,
        )
        rows.append(evaluate(model, X.iloc[test_idx].values, y.iloc[test_idx].values, scaler))
    df = pd.DataFrame(rows)
    return {
        "cv_accuracy_mean": df["accuracy"].mean(),
        "cv_accuracy_std": df["accuracy"].std(),
        "cv_auc_mean": df["auc"].mean(),
        "cv_auc_std": df["auc"].std(),
    }


def roc_points(model, X_test: np.ndarray, y_test: np.ndarray, scaler=None):
    Xt = scaler.transform(X_test) if scaler else X_test
    prob = model.predict_proba(Xt)[:, 1]
    return roc_curve(y_test, prob)
