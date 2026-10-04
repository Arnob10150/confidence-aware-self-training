from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import FIGURES_DIR, RATIOS, SEEDS, ensure_dirs, get_splits, load_processed, make_models, roc_points, self_train


PRIMARY = "with_score"


def _mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    return grouped


def build_summary_tables() -> pd.DataFrame:
    frames = []
    baseline = pd.read_csv("results/baseline_results.csv")
    baseline["experiment"] = "Baseline"
    frames.append(
        baseline.rename(columns={"model": "method"}).assign(variant="supervised_only")
    )

    selftraining = pd.read_csv("results/selftraining_results.csv")
    selftraining["experiment"] = "SelfTraining"
    selftraining["method"] = selftraining["model"] + "_" + selftraining["variant"]
    frames.append(selftraining)

    ssl = pd.read_csv("results/ssl_comparison.csv")
    ssl["experiment"] = "SSLComparison"
    ssl["variant"] = ssl["method"]
    frames.append(ssl)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    summary = _mean_std(
        combined,
        ["config", "experiment", "method", "variant", "ratio"],
        ["accuracy", "precision", "recall", "f1", "auc"],
    )
    summary.to_csv("results/summary_table.csv", index=False)
    return summary


def plot_baseline_vs_selftraining():
    baseline = pd.read_csv("results/baseline_results.csv")
    st = pd.read_csv("results/selftraining_results.csv")
    base = baseline[baseline["config"] == PRIMARY].copy()
    dyn = st[(st["config"] == PRIMARY) & (st["variant"] == "dynamic")].copy()
    base["family"] = "Baseline"
    dyn["family"] = "Dynamic Self-Training"
    plot_df = pd.concat(
        [
            base[["ratio", "model", "family", "accuracy", "auc"]],
            dyn[["ratio", "model", "family", "accuracy", "auc"]],
        ],
        ignore_index=True,
    )
    for metric in ["accuracy", "auc"]:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=plot_df, x="ratio", y=metric, hue="model", style="family", marker="o", errorbar="sd")
        plt.title(f"{metric.upper()} vs Labeled Ratio")
        plt.xlabel("Labeled training ratio")
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{metric}_vs_labeled_ratio.png", dpi=200)
        plt.close()


def plot_threshold_comparison():
    st = pd.read_csv("results/selftraining_results.csv")
    df = st[(st["config"] == PRIMARY) & (st["model"] == "XGB")]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="ratio", y="auc", hue="variant", errorbar="sd")
    plt.title("Confidence Threshold Strategy Comparison (XGB)")
    plt.xlabel("Labeled training ratio")
    plt.ylabel("ROC-AUC")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_variant_auc.png", dpi=200)
    plt.close()


def plot_ssl_comparison():
    ssl = pd.read_csv("results/ssl_comparison.csv")
    df = ssl[ssl["config"] == PRIMARY]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="ratio", y="auc", hue="method", errorbar="sd")
    plt.title("SSL Method Comparison")
    plt.xlabel("Labeled training ratio")
    plt.ylabel("ROC-AUC")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ssl_method_auc.png", dpi=200)
    plt.close()


def plot_error_propagation():
    err = pd.read_csv("results/error_propagation.csv")
    df = err[err["config"] == PRIMARY]
    for metric, filename, ylabel in [
        ("pl_acc", "error_propagation_pl_acc.png", "Pseudo-label accuracy"),
        ("cumulative_wrong", "noise_injection_cumulative_wrong.png", "Cumulative wrong pseudo-labels"),
        ("cumulative_error_rate", "cumulative_error_rate.png", "Cumulative error rate"),
    ]:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df, x="iter", y=metric, hue="variant", marker="o", errorbar="sd")
        plt.title(ylabel + " Across Self-Training Iterations")
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=200)
        plt.close()


def plot_roc_curves():
    X, y = load_processed(PRIMARY)
    xgb_spec = make_models()["XGB"]
    plt.figure(figsize=(7, 6))
    for ratio in RATIOS:
        auc_values = []
        fpr_plot = tpr_plot = None
        for seed in SEEDS:
            X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)
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
                seed=seed,
            )
            fpr, tpr, _thr = roc_points(model, X_test, y_test, scaler)
            if seed == 42:
                fpr_plot, tpr_plot = fpr, tpr
            from sklearn.metrics import auc

            auc_values.append(auc(fpr, tpr))
        plt.plot(fpr_plot, tpr_plot, label=f"{int(ratio * 100)}% labels (AUC {sum(auc_values)/len(auc_values):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Dynamic Self-Training XGB")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_dynamic_xgb_by_ratio.png", dpi=200)
    plt.close()


def plot_dataset_distribution():
    summary = pd.read_csv("results/dataset_summary.csv")
    counts = pd.DataFrame(
        {
            "class": ["Non-HIGH", "HIGH"],
            "count": [int(summary["non_high_count"].iloc[0]), int(summary["high_count"].iloc[0])],
        }
    )
    plt.figure(figsize=(5, 4))
    sns.barplot(data=counts, x="class", y="count")
    plt.title("Binary CVD Risk Target Distribution")
    plt.xlabel("Class")
    plt.ylabel("Patient records")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=200)
    plt.close()


def plot_workflow_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    boxes = [
        (0.02, 0.58, "CAIR-CVD-2025\n1,529 records"),
        (0.19, 0.58, "Preprocess\nencode + impute\nwith/without score"),
        (0.37, 0.58, "Label Scarcity\n1, 5, 10, 20, 40%"),
        (0.56, 0.58, "Models\nLR, SVM, RF, XGB"),
        (0.75, 0.58, "Evaluation\nAcc, F1, AUC"),
        (0.20, 0.18, "Confidence-Aware\nSelf-Training\n0.80 / 0.95 / dynamic"),
        (0.47, 0.18, "SSL Comparison\nMean Teacher\nLabel Propagation"),
        (0.73, 0.18, "Diagnostics\nError propagation\nSHAP explanations"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), 0.15, 0.22, fill=True, edgecolor="#2b4c7e", facecolor="#eaf1fb", linewidth=1.5)
        )
        ax.text(x + 0.075, y + 0.11, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.17, 0.69), (0.19, 0.69)),
        ((0.34, 0.69), (0.37, 0.69)),
        ((0.52, 0.69), (0.56, 0.69)),
        ((0.71, 0.69), (0.75, 0.69)),
        ((0.445, 0.58), (0.275, 0.40)),
        ((0.635, 0.58), (0.545, 0.40)),
        ((0.825, 0.58), (0.805, 0.40)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4))
    ax.set_title("End-to-End Experimental Workflow", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "workflow_diagram.png", dpi=200)
    plt.close()


def run_visualizations() -> pd.DataFrame:
    ensure_dirs()
    summary = build_summary_tables()
    plot_workflow_diagram()
    plot_dataset_distribution()
    plot_baseline_vs_selftraining()
    plot_threshold_comparison()
    plot_ssl_comparison()
    plot_error_propagation()
    plot_roc_curves()
    print("Saved visualizations and results/summary_table.csv", summary.shape)
    return summary


if __name__ == "__main__":
    run_visualizations()
