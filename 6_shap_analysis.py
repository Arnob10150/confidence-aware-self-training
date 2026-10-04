from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils import SHAP_DIR, ensure_dirs, get_splits, load_processed, make_models, self_train


def _shap_values(explainer, X):
    values = explainer.shap_values(X)
    if isinstance(values, list):
        return values[1]
    if getattr(values, "ndim", 0) == 3:
        return values[:, :, 1]
    return values


def run_shap() -> pd.DataFrame:
    ensure_dirs()
    rows = []
    config = "framingham"
    X, y = load_processed(config)
    feature_names = X.columns.tolist()
    xgb_spec = make_models()["XGB"]

    for ratio in [0.05, 0.40]:
        X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed=42)
        model, scaler, _log = self_train(
            xgb_spec.estimator_factory,
            X_l,
            y_l,
            X_u,
            y_u,
            X_test,
            y_test,
            "dynamic",
            scale=xgb_spec.scale,
            seed=42,
        )
        Xt = scaler.transform(X_test) if scaler else X_test
        sample_size = min(250, len(Xt))
        X_sample = Xt[:sample_size]
        explainer = shap.TreeExplainer(model)
        sv = _shap_values(explainer, X_sample)

        suffix = f"{int(ratio * 100)}pct"
        shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"summary_{suffix}.pdf", bbox_inches="tight")
        plt.close()

        shap.summary_plot(
            sv,
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=15,
        )
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"bar_{suffix}.pdf", bbox_inches="tight")
        plt.close()

        high_idx = int(np.where(y_test == 1)[0][0])
        high_x = Xt[high_idx : high_idx + 1]
        high_sv = _shap_values(explainer, high_x)
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = np.asarray(base).ravel()[-1]
        explanation = shap.Explanation(
            values=high_sv[0],
            base_values=base,
            data=high_x[0],
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"waterfall_high_{suffix}.pdf", bbox_inches="tight")
        plt.close()

        mean_abs = np.abs(sv).mean(axis=0)
        for feature, value in zip(feature_names, mean_abs):
            rows.append(
                {
                    "config": config,
                    "ratio": ratio,
                    "feature": feature,
                    "mean_abs_shap": value,
                }
            )
    out = pd.DataFrame(rows).sort_values(["ratio", "mean_abs_shap"], ascending=[True, False])
    out.to_csv("results/shap_importance.csv", index=False)
    print("Saved SHAP figures and results/shap_importance.csv", out.shape)
    return out


if __name__ == "__main__":
    run_shap()
