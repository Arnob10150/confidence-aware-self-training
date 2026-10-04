from __future__ import annotations

import pandas as pd

from utils import SEEDS, VARIANTS, ensure_dirs, get_splits, load_processed, make_models, self_train


def run_imbalance_analysis() -> pd.DataFrame:
    ensure_dirs()
    rows = []
    X, y = load_processed("framingham")
    xgb_spec = make_models()["XGB"]
    for variant in VARIANTS:
        for seed in SEEDS:
            X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, 0.05, seed)
            _model, _scaler, log = self_train(
                xgb_spec.estimator_factory,
                X_l,
                y_l,
                X_u,
                y_u,
                X_test,
                y_test,
                variant,
                scale=xgb_spec.scale,
                seed=seed,
            )
            for item in log:
                rows.append({"config": "framingham", "ratio": 0.05, "seed": seed, "variant": variant, **item})
    out = pd.DataFrame(rows)
    out.to_csv("results/imbalance_analysis.csv", index=False)
    print("Saved results/imbalance_analysis.csv", out.shape)
    return out


if __name__ == "__main__":
    run_imbalance_analysis()
