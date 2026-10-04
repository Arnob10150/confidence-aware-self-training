from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # Type 42 (TrueType), not Type 3, for IEEE PDF eXpress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import FIGURES_DIR

PRIMARY = "framingham"
DISPLAY_NAMES = {"vanilla": "Standard", "static_high": "Static High", "dynamic": "Dynamic"}
SSL_DISPLAY_NAMES = {
    "SelfTrainingDynamic": "Dynamic Self-Training",
    "MeanTeacher": "Mean Teacher",
    "LabelPropagation": "Label Propagation",
}


def _curve(ax, x, mean, color, label, linestyle, marker):
    ax.plot(x, mean, color=color, linestyle=linestyle, marker=marker, markersize=4.5, linewidth=1.8, label=label)


def combine_accuracy_panels():
    st = pd.read_csv("results/selftraining_results.csv")
    baseline = pd.read_csv("results/baseline_results.csv")
    dyn = st[(st["config"] == PRIMARY) & (st["variant"] == "dynamic")].copy()
    base = baseline[baseline["config"] == PRIMARY].copy()

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.6))

    styles_a = {"LR": ("#1D9E75", "-", "s"), "SVM": ("#D85A30", "--", "^"), "RF": ("#7A4EB3", "-.", "D"), "XGB": ("#185FA5", "-", "o")}
    for model_name, (color, ls, marker) in styles_a.items():
        grouped = dyn[dyn["model"] == model_name].groupby("ratio")["accuracy"].mean()
        _curve(axes[0], grouped.index.to_numpy(), grouped.to_numpy(), color, model_name, ls, marker)
    axes[0].set_xlabel("Labeled training ratio", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_title("(a) Accuracy by model", fontsize=11)
    axes[0].tick_params(labelsize=10)
    axes[0].margins(y=0.1)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=4, fontsize=9, frameon=False, columnspacing=1.2, handletextpad=0.4)
    axes[0].grid(alpha=0.12)

    styles_b = {"Supervised": ("#1D9E75", "-", "s"), "Dynamic self-training": ("#185FA5", "--", "o")}
    curves_b = {
        "Supervised": base.groupby("ratio")["accuracy"].mean(),
        "Dynamic self-training": dyn.groupby("ratio")["accuracy"].mean(),
    }
    for label, grouped in curves_b.items():
        color, ls, marker = styles_b[label]
        _curve(axes[1], grouped.index.to_numpy(), grouped.to_numpy(), color, label, ls, marker)
    axes[1].set_xlabel("Labeled training ratio", fontsize=11)
    axes[1].set_ylabel("Mean accuracy", fontsize=11)
    axes[1].set_title("(b) Supervised vs. self-trained", fontsize=11)
    axes[1].tick_params(labelsize=10)
    axes[1].margins(y=0.1)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=9, frameon=False, columnspacing=1.2, handletextpad=0.4)
    axes[1].grid(alpha=0.12)

    fig.subplots_adjust(hspace=0.95)
    plt.savefig(FIGURES_DIR / "accuracy_combined.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close()


def combine_threshold_ssl_panels():
    st = pd.read_csv("results/selftraining_results.csv")
    ssl = pd.read_csv("results/ssl_comparison.csv")
    df = st[(st["config"] == PRIMARY) & (st["model"] == "XGB")].copy()
    df["labeled_ratio"] = df["ratio"].map(lambda v: f"{int(v * 100)}%")
    df["variant"] = df["variant"].map(lambda v: DISPLAY_NAMES.get(v, v))
    dfs = ssl[ssl["config"] == PRIMARY].copy()
    dfs["labeled_ratio"] = dfs["ratio"].map(lambda v: f"{int(v * 100)}%")
    dfs["method"] = dfs["method"].map(lambda v: SSL_DISPLAY_NAMES.get(v, v))

    import seaborn as sns

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.8))
    sns.barplot(
        data=df, y="labeled_ratio", x="auc", hue="variant",
        hue_order=["Standard", "Static High", "Dynamic"], errorbar=None, orient="h", ax=axes[0],
    )
    axes[0].set_xlabel("ROC-AUC", fontsize=11)
    axes[0].set_ylabel("Labeled ratio", fontsize=11)
    axes[0].set_xlim(0.55, 0.73)
    axes[0].set_title("(a) Threshold strategy (XGB)", fontsize=11)
    axes[0].tick_params(labelsize=10)
    axes[0].legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3, fontsize=9, frameon=False)

    sns.barplot(
        data=dfs, y="labeled_ratio", x="auc", hue="method",
        hue_order=["Dynamic Self-Training", "Mean Teacher", "Label Propagation"], errorbar=None, orient="h", ax=axes[1],
    )
    axes[1].set_xlabel("ROC-AUC", fontsize=11)
    axes[1].set_ylabel("Labeled ratio", fontsize=11)
    axes[1].set_xlim(0.48, 0.73)
    axes[1].set_title("(b) SSL method comparison", fontsize=11)
    axes[1].tick_params(labelsize=10)
    axes[1].legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, fontsize=9, frameon=False)

    fig.subplots_adjust(hspace=0.95)
    plt.savefig(FIGURES_DIR / "threshold_ssl_combined.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close()


def combine_error_panels():
    err = pd.read_csv("results/error_propagation.csv")
    df = err[err["config"] == PRIMARY]
    styles = {"vanilla": ("#D85A30", "--", "^"), "static_high": ("#7A4EB3", "-.", "D"), "dynamic": ("#185FA5", "-", "o")}

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.6))
    for metric, ax, ylabel, title in [
        ("pl_acc", axes[0], "Pseudo-label accuracy", "(a) PL accuracy by iteration"),
        ("cumulative_error_rate", axes[1], "Cumulative error rate", "(b) Cumulative error rate"),
    ]:
        for variant, (color, ls, marker) in styles.items():
            grouped = df[df["variant"] == variant].groupby("iter")[metric].mean()
            if grouped.empty:
                continue
            _curve(ax, grouped.index.to_numpy(), grouped.to_numpy(), color, DISPLAY_NAMES.get(variant, variant), ls, marker)
        ax.set_xlabel("Self-training iteration", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.margins(y=0.1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3, fontsize=9, frameon=False, columnspacing=1.0, handletextpad=0.4)
        ax.grid(alpha=0.12)

    fig.subplots_adjust(hspace=0.95)
    plt.savefig(FIGURES_DIR / "error_combined.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close()


def combine_shap_panels():
    shap = pd.read_csv("results/shap_importance.csv")
    fig, axes = plt.subplots(2, 1, figsize=(3.8, 5.2))
    for ratio, ax, title in [(0.05, axes[0], "(a) 5% labels"), (0.40, axes[1], "(b) 40% labels")]:
        sub = shap[shap["ratio"] == ratio].sort_values("mean_abs_shap", ascending=True).tail(8)
        ax.barh(sub["feature"], sub["mean_abs_shap"], color="#185FA5")
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(title, fontsize=11, pad=10)
        ax.tick_params(axis="both", labelsize=10)
        ax.margins(x=0.08)
    fig.subplots_adjust(hspace=0.55)
    plt.savefig(FIGURES_DIR / "shap_combined.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close()


if __name__ == "__main__":
    combine_accuracy_panels()
    combine_threshold_ssl_panels()
    combine_error_panels()
    combine_shap_panels()
    print("Saved combined figures")
