from __future__ import annotations

import pandas as pd

from utils import CONFIGS, SEEDS, VARIANTS, ensure_dirs, get_splits, load_processed, make_models, self_train


def run_error_propagation() -> pd.DataFrame:
    ensure_dirs()
    rows = []
    xgb_spec = make_models()["XGB"]
    for config in CONFIGS:
        X, y = load_processed(config)
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
                    rows.append({"config": config, "ratio": 0.05, "seed": seed, "variant": variant, **item})
    out = pd.DataFrame(rows)
    out.to_csv("results/error_propagation.csv", index=False)
    print("Saved results/error_propagation.csv", out.shape)
    return out


if __name__ == "__main__":
    run_error_propagation()
