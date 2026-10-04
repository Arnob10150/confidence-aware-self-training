from __future__ import annotations

import pandas as pd

from utils import (
    CONFIGS,
    RATIOS,
    SEEDS,
    VARIANTS,
    ensure_dirs,
    evaluate,
    get_splits,
    load_processed,
    make_models,
    self_train,
)


def run_self_training() -> pd.DataFrame:
    ensure_dirs()
    rows = []
    log_rows = []
    models = make_models()
    for config in CONFIGS:
        X, y = load_processed(config)
        for ratio in RATIOS:
            for seed in SEEDS:
                X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)
                for model_name, spec in models.items():
                    for variant in VARIANTS:
                        model, scaler, log = self_train(
                            spec.estimator_factory,
                            X_l,
                            y_l,
                            X_u,
                            y_u,
                            X_test,
                            y_test,
                            variant,
                            scale=spec.scale,
                            seed=seed,
                        )
                        metrics = evaluate(model, X_test, y_test, scaler)
                        rows.append(
                            {
                                "config": config,
                                "ratio": ratio,
                                "seed": seed,
                                "model": model_name,
                                "variant": variant,
                                "iterations": len(log),
                                "pseudo_added_total": sum(item["added"] for item in log),
                                **metrics,
                            }
                        )
                        for item in log:
                            log_rows.append(
                                {
                                    "config": config,
                                    "ratio": ratio,
                                    "seed": seed,
                                    "model": model_name,
                                    "variant": variant,
                                    **item,
                                }
                            )
    out = pd.DataFrame(rows)
    logs = pd.DataFrame(log_rows)
    out.to_csv("results/selftraining_results.csv", index=False)
    logs.to_csv("results/selftraining_iteration_logs.csv", index=False)
    print("Saved results/selftraining_results.csv", out.shape)
    print("Saved results/selftraining_iteration_logs.csv", logs.shape)
    return out


if __name__ == "__main__":
    run_self_training()
