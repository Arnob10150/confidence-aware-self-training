from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import CONFIGS, RATIOS, SEEDS, ensure_dirs, evaluate, fit_model, get_splits, load_processed, make_models


def run_baselines() -> pd.DataFrame:
    ensure_dirs()
    rows = []
    ceiling_rows = []
    models = make_models()
    for config in CONFIGS:
        X, y = load_processed(config)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, spec in models.items():
            fold_rows = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
                model, scaler = fit_model(
                    spec.estimator_factory(42),
                    X.iloc[train_idx].values,
                    y.iloc[train_idx].values,
                    spec.scale,
                )
                fold_rows.append(evaluate(model, X.iloc[test_idx].values, y.iloc[test_idx].values, scaler))
                ceiling_rows.append(
                    {
                        "config": config,
                        "model": name,
                        "fold": fold,
                        **fold_rows[-1],
                    }
                )
        for ratio in RATIOS:
            for seed in SEEDS:
                X_l, y_l, _X_u, _y_u, X_test, y_test = get_splits(X, y, ratio, seed)
                for name, spec in models.items():
                    model, scaler = fit_model(spec.estimator_factory(seed), X_l, y_l, spec.scale)
                    metrics = evaluate(model, X_test, y_test, scaler)
                    rows.append(
                        {
                            "config": config,
                            "ratio": ratio,
                            "seed": seed,
                            "model": name,
                            **metrics,
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv("results/baseline_results.csv", index=False)
    ceiling = pd.DataFrame(ceiling_rows)
    ceiling.to_csv("results/cv_ceiling.csv", index=False)
    print("Saved results/baseline_results.csv", out.shape)
    print("Saved results/cv_ceiling.csv", ceiling.shape)
    return out


if __name__ == "__main__":
    run_baselines()
