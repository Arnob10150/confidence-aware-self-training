from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # Type 42 (TrueType), not Type 3, for IEEE PDF eXpress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import FIGURES_DIR, RATIOS, SEEDS, ensure_dirs, get_splits, load_processed, make_models, roc_points, self_train


PRIMARY = "framingham"

DISPLAY_NAMES = {"vanilla": "Standard", "static_high": "Static High", "dynamic": "Dynamic"}
SSL_DISPLAY_NAMES = {
    "SelfTrainingDynamic": "Dynamic Self-Training",
    "MeanTeacher": "Mean Teacher",
    "LabelPropagation": "Label Propagation",
}


def _mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    return grouped


def _plot_curve(ax, x, mean, std, color, label, linestyle, marker):
    mean = np.asarray(mean, dtype=float)
    ax.plot(
        x,
        mean,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=7,
        linewidth=2.6,
        label=label,
        zorder=3,
    )


def _finish_curve(ax, ylabel: str, values: list[float], filename: str, xlabel: str = "Labeled training ratio"):
    ymin = max(0.0, min(values) - 0.03)
    ymax = min(1.0, max(values) + 0.03)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.12)
    n_items = len(ax.get_legend_handles_labels()[1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(n_items, 2), fontsize=12, frameon=False)
    fig = ax.get_figure()
    fig.subplots_adjust(bottom=0.32)
    plt.savefig(FIGURES_DIR / filename, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close()


def build_summary_tables() -> pd.DataFrame:
    frames = []
    baseline = pd.read_csv("results/baseline_results.csv")
    baseline["experiment"] = "Baseline"
    frames.append(baseline.rename(columns={"model": "method"}).assign(variant="supervised_only"))

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


def plot_xgb_curves():
    baseline = pd.read_csv("results/baseline_results.csv")
    st = pd.read_csv("results/selftraining_results.csv")
    base = baseline[(baseline["config"] == PRIMARY) & (baseline["model"] == "XGB")].copy()
    base["method"] = "Supervised XGB"
    variants = st[(st["config"] == PRIMARY) & (st["model"] == "XGB")].copy()
    variants["method"] = variants["variant"].map(
        {
            "vanilla": "Self-training 0.80",
            "static_high": "Self-training 0.95",
            "dynamic": "Dynamic self-training",
        }
    )
    plot_df = pd.concat(
        [
            base[["ratio", "method", "accuracy", "precision", "recall", "f1", "auc"]],
            variants[["ratio", "method", "accuracy", "precision", "recall", "f1", "auc"]],
        ],
        ignore_index=True,
    )
    styles = {
        "Supervised XGB": ("#1D9E75", "-", "s"),
        "Self-training 0.80": ("#D85A30", "--", "^"),
        "Self-training 0.95": ("#7A4EB3", "-.", "D"),
        "Dynamic self-training": ("#185FA5", "-", "o"),
    }
    for metric, ylabel, filename in [
        ("auc", "ROC-AUC", "auc_vs_labeled_ratio.pdf"),
        ("f1", "F1-score", "f1_vs_labeled_ratio.pdf"),
        ("recall", "Recall", "recall_vs_labeled_ratio.pdf"),
        ("accuracy", "Accuracy", "accuracy_vs_labeled_ratio.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        values = []
        for method, (color, linestyle, marker) in styles.items():
            grouped = plot_df[plot_df["method"] == method].groupby("ratio")[metric].agg(["mean", "std"])
            x = grouped.index.to_numpy()
            mean = grouped["mean"].to_numpy()
            std = grouped["std"].fillna(0).to_numpy()
            values.extend(mean.tolist())
            _plot_curve(ax, x, mean, std, color, method, linestyle, marker)
        _finish_curve(ax, ylabel, values, filename)


def plot_model_accuracy_variation():
    st = pd.read_csv("results/selftraining_results.csv")
    df = st[(st["config"] == PRIMARY) & (st["variant"] == "dynamic")].copy()
    styles = {
        "LR": ("#1D9E75", "-", "s"),
        "SVM": ("#D85A30", "--", "^"),
        "RF": ("#7A4EB3", "-.", "D"),
        "XGB": ("#185FA5", "-", "o"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    values = []
    for model_name, (color, linestyle, marker) in styles.items():
        grouped = df[df["model"] == model_name].groupby("ratio")["accuracy"].mean()
        x = grouped.index.to_numpy()
        mean = grouped.to_numpy()
        values.extend(mean.tolist())
        _plot_curve(ax, x, mean, np.zeros_like(mean), color, model_name, linestyle, marker)
    _finish_curve(ax, "Accuracy", values, "accuracy_variation_models.pdf")


def plot_supervised_vs_selftrained_accuracy():
    baseline = pd.read_csv("results/baseline_results.csv")
    st = pd.read_csv("results/selftraining_results.csv")
    base = baseline[baseline["config"] == PRIMARY].copy()
    dyn = st[(st["config"] == PRIMARY) & (st["variant"] == "dynamic")].copy()
    curves = {
        "Supervised": base.groupby("ratio")["accuracy"].mean(),
        "Dynamic self-training": dyn.groupby("ratio")["accuracy"].mean(),
    }
    styles = {
        "Supervised": ("#1D9E75", "-", "s"),
        "Dynamic self-training": ("#185FA5", "--", "o"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    values = []
    for label, grouped in curves.items():
        color, linestyle, marker = styles[label]
        x = grouped.index.to_numpy()
        mean = grouped.to_numpy()
        values.extend(mean.tolist())
        _plot_curve(ax, x, mean, np.zeros_like(mean), color, label, linestyle, marker)
    _finish_curve(ax, "Mean accuracy across classifiers", values, "performance_supervised_vs_selftrained.pdf")


def plot_threshold_comparison():
    st = pd.read_csv("results/selftraining_results.csv")
    df = st[(st["config"] == PRIMARY) & (st["model"] == "XGB")].copy()
    df["labeled_ratio"] = df["ratio"].map(lambda value: f"{int(value * 100)}% labels")
    df["variant"] = df["variant"].map(lambda v: DISPLAY_NAMES.get(v, v))
    plt.figure(figsize=(8, 5.2))
    sns.barplot(
        data=df,
        y="labeled_ratio",
        x="auc",
        hue="variant",
        hue_order=["Standard", "Static High", "Dynamic"],
        errorbar=None,
        orient="h",
    )
    plt.xlabel("ROC-AUC")
    plt.ylabel("Labeled training ratio")
    plt.xlim(0.55, 0.73)
    plt.legend(title="Threshold strategy", frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_variant_auc.pdf")
    plt.close()


def plot_ssl_comparison():
    ssl = pd.read_csv("results/ssl_comparison.csv")
    df = ssl[ssl["config"] == PRIMARY].copy()
    df["labeled_ratio"] = df["ratio"].map(lambda value: f"{int(value * 100)}% labels")
    df["method"] = df["method"].map(lambda v: SSL_DISPLAY_NAMES.get(v, v))
    plt.figure(figsize=(8, 5.2))
    sns.barplot(
        data=df,
        y="labeled_ratio",
        x="auc",
        hue="method",
        hue_order=["Dynamic Self-Training", "Mean Teacher", "Label Propagation"],
        errorbar=None,
        orient="h",
    )
    plt.xlabel("ROC-AUC")
    plt.ylabel("Labeled training ratio")
    plt.xlim(0.48, 0.73)
    plt.legend(title="Method", frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ssl_method_auc.pdf")
    plt.close()


def plot_error_propagation():
    err = pd.read_csv("results/error_propagation.csv")
    df = err[err["config"] == PRIMARY]
    styles = {
        "vanilla": ("#D85A30", "--", "^"),
        "static_high": ("#7A4EB3", "-.", "D"),
        "dynamic": ("#185FA5", "-", "o"),
    }
    for metric, filename, ylabel in [
        ("pl_acc", "error_propagation_pl_acc.pdf", "Pseudo-label accuracy"),
        ("cumulative_wrong", "noise_injection_cumulative_wrong.pdf", "Cumulative wrong pseudo-labels"),
        ("cumulative_error_rate", "cumulative_error_rate.pdf", "Cumulative error rate"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        values = []
        for variant, (color, linestyle, marker) in styles.items():
            grouped = df[df["variant"] == variant].groupby("iter")[metric].agg(["mean", "std"])
            if grouped.empty:
                continue
            x = grouped.index.to_numpy()
            mean = grouped["mean"].to_numpy()
            std = grouped["std"].fillna(0).to_numpy()
            values.extend(mean.tolist())
            _plot_curve(ax, x, mean, std, color, DISPLAY_NAMES.get(variant, variant), linestyle, marker)
        _finish_curve(ax, ylabel, values, filename, xlabel="Self-training iteration")


def plot_imbalance_analysis():
    imb = pd.read_csv("results/imbalance_analysis.csv")
    styles = {
        "vanilla": ("#D85A30", "--", "^"),
        "static_high": ("#7A4EB3", "-.", "D"),
        "dynamic": ("#185FA5", "-", "o"),
    }
    for metric, filename, ylabel in [
        ("pl_pos_rate", "pseudo_label_positive_rate.pdf", "Pseudo-label positive rate"),
        ("cumulative_pos_rate", "cumulative_pseudo_label_positive_rate.pdf", "Cumulative pseudo-label positive rate"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        values = []
        for variant, (color, linestyle, marker) in styles.items():
            grouped = imb[imb["variant"] == variant].groupby("iter")[metric].agg(["mean", "std"])
            if grouped.empty:
                continue
            x = grouped.index.to_numpy()
            mean = grouped["mean"].to_numpy()
            std = grouped["std"].fillna(0).to_numpy()
            values.extend(mean.tolist())
            _plot_curve(ax, x, mean, std, color, DISPLAY_NAMES.get(variant, variant), linestyle, marker)
        ax.axhline(0.152, color="#333333", linewidth=1.0, alpha=0.5, label="True positive rate")
        values.append(0.152)
        _finish_curve(ax, ylabel, values, filename, xlabel="Self-training iteration")


def plot_roc_curves(model_name: str = "XGB", filename: str = "roc_dynamic_xgb_by_ratio.pdf"):
    X, y = load_processed(PRIMARY)
    spec = make_models()[model_name]
    plt.figure(figsize=(7, 6))
    for ratio in RATIOS:
        auc_values = []
        fpr_plot = tpr_plot = None
        for seed in SEEDS:
            X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)
            model, scaler, _log = self_train(
                spec.estimator_factory,
                X_l,
                y_l,
                X_u,
                y_u,
                X_test,
                y_test,
                "dynamic",
                scale=spec.scale,
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
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()


def plot_dataset_distribution():
    summary = pd.read_csv("results/dataset_summary.csv")
    counts = pd.DataFrame(
        {
            "class": ["No 10-year CHD", "10-year CHD"],
            "count": [int(summary["negative_count"].iloc[0]), int(summary["positive_count"].iloc[0])],
        }
    )
    plt.figure(figsize=(5.5, 4))
    sns.barplot(data=counts, x="class", y="count")
    plt.xlabel("Class")
    plt.ylabel("Patient records")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.pdf")
    plt.close()


def plot_workflow_diagram():
    from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(7.2, 9.4))
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 20.0)

    C = {
        "band": ("#cfead0", "#3f7a47"),
        "data": ("#c7e7ee", "#2b8090"),
        "pre": ("#cfe5f6", "#2b6cb0"),
        "dia": ("#d6edd6", "#3f7a47"),
        "scale": ("#fde2c4", "#c0772b"),
        "clf1": ("#cfe8d4", "#3f7a47"),
        "clf2": ("#c9e6f2", "#2b7fa8"),
        "clf3": ("#fde2c4", "#c0772b"),
        "clf4": ("#e7dcf3", "#6b46c1"),
        "ssl": ("#e7dcf3", "#6b46c1"),
        "diag": ("#cfe5f6", "#2b6cb0"),
        "metric": ("#e8edf2", "#54657a"),
        "final": ("#fdeeb3", "#b8901f"),
    }

    def box(cx, cy, w, h, lines, key, fs=11.5, bold_first=True):
        face, edge = C[key]
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     linewidth=1.5, edgecolor=edge, facecolor=face, zorder=3))
        if isinstance(lines, str):
            lines = [lines]
        if bold_first and len(lines) > 1:
            ax.text(cx, cy + 0.02 + (len(lines) - 1) * 0.16 * fs / 11.5, lines[0],
                    ha="center", va="center", fontsize=fs, fontweight="bold", color=edge, zorder=4)
            ax.text(cx, cy - 0.30, "\n".join(lines[1:]), ha="center", va="center",
                    fontsize=fs - 1.5, color="#1a202c", zorder=4, linespacing=1.35)
        else:
            ax.text(cx, cy, "\n".join(lines), ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold_first else "normal",
                    color=edge if bold_first else "#1a202c", zorder=4, linespacing=1.3)

    def cylinder(cx, cy, w, h, lines, key, fs=11):
        face, edge = C[key]
        eh = h * 0.26
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + eh / 2), w, h - eh,
                     facecolor=face, edgecolor="none", zorder=3))
        ax.add_patch(Ellipse((cx, cy - h / 2 + eh / 2), w, eh, facecolor=face, edgecolor=edge, lw=1.5, zorder=3))
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + eh / 2), w, h - eh,
                     facecolor="none", edgecolor=edge, lw=1.5, zorder=3))
        ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2 + eh / 2, cy + h / 2 - eh / 2], color=edge, lw=1.5, zorder=3)
        ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2 + eh / 2, cy + h / 2 - eh / 2], color=edge, lw=1.5, zorder=3)
        ax.add_patch(Ellipse((cx, cy + h / 2 - eh / 2), w, eh, facecolor=face, edgecolor=edge, lw=1.5, zorder=4))
        ax.text(cx, cy + h / 2 - eh / 2, lines[0], ha="center", va="center", fontsize=fs,
                fontweight="bold", color=edge, zorder=5)
        ax.text(cx, cy - eh / 2, "\n".join(lines[1:]), ha="center", va="center", fontsize=fs - 1.5,
                color="#1a202c", zorder=5, linespacing=1.3)

    def diamond(cx, cy, w, h, lines, key, fs=11):
        face, edge = C[key]
        ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                     closed=True, facecolor=face, edgecolor=edge, lw=1.5, zorder=3))
        ax.text(cx, cy, "\n".join(lines), ha="center", va="center", fontsize=fs,
                fontweight="bold", color=edge, zorder=4, linespacing=1.25)

    def band(y0, y1, label):
        face, edge = C["band"]
        ax.add_patch(FancyBboxPatch((0.15, y0), 0.95, y1 - y0,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     linewidth=1.3, edgecolor=edge, facecolor=face, zorder=2))
        ax.text(0.62, (y0 + y1) / 2, label, ha="center", va="center", rotation=90,
                fontsize=12, fontweight="bold", color=edge, zorder=4)

    def arrow(x1, y1, x2, y2, rad=0.0):
        style = "arc3,rad=%.2f" % rad
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#2d3748", lw=1.6,
                                    shrinkA=1, shrinkB=1, connectionstyle=style), zorder=1)

    def cap(cx, cy, text):
        ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5, fontstyle="italic", color="#444", zorder=4)

    cM = 5.7  # main column center

    # bands
    band(16.7, 19.4, "Data & Preprocessing")
    band(9.1, 15.6, "Modeling")
    band(0.4, 8.2, "Evaluation & Output")

    # Row 1: data cylinder
    cylinder(cM, 18.4, 4.6, 1.7, ["Data Collection", "Framingham 4,240 × 15", "15.2% CHD-positive"], "data")
    # Row 2: preprocessing (two boxes)
    box(3.7, 16.0, 3.4, 1.5, ["Cleaning", "median imputation", "(no row drop)"], "pre", fs=11)
    box(7.7, 16.0, 3.4, 1.5, ["Train–Test Split", "stratified 80/20"], "pre", fs=11)
    # Row 3: diamond label scarcity
    diamond(cM, 13.6, 4.6, 2.0, ["Label-Scarcity", "Sampling", "1/5/10/20/40%"], "dia", fs=11)
    # Row 4: scaling
    box(3.7, 11.4, 3.4, 1.4, ["StandardScaler", "LR · SVM"], "scale", fs=11)
    box(7.7, 11.4, 3.4, 1.4, ["Raw Features", "RF · XGB"], "scale", fs=11)
    # Row 5: classifiers (four boxes) + caption
    cys = 9.7
    box(2.7, cys, 1.7, 1.1, ["LR"], "clf1", fs=12, bold_first=False)
    box(4.55, cys, 1.7, 1.1, ["SVM"], "clf2", fs=12, bold_first=False)
    box(6.4, cys, 1.7, 1.1, ["RF"], "clf3", fs=12, bold_first=False)
    box(8.25, cys, 1.7, 1.1, ["XGB"], "clf4", fs=12, bold_first=False)
    cap(5.5, 8.85, "Class-Balanced Base Classifiers")
    # Row 6: self-training (left) + SSL baselines cluster (right)
    box(3.5, 6.9, 3.6, 1.7, ["Confidence", "Self-Training", r"$\tau$=0.80/0.95/dyn"], "ssl", fs=11)
    box(8.3, 7.35, 2.4, 1.0, ["Mean Teacher"], "ssl", fs=10.5, bold_first=False)
    box(11.9, 7.35, 3.2, 1.0, ["Label Propagation"], "ssl", fs=10.5, bold_first=False)
    cap(10.1, 6.55, "SSL Baselines (surr. + deep MT)")
    # Row 7: pseudo-label diagnostics (merge)
    box(cM, 4.8, 7.4, 1.4, ["Pseudo-Label Diagnostics", "accuracy · positive rate · error propagation"], "diag", fs=11)
    # Row 8: evaluating diamond + metric/calibration cluster
    diamond(3.5, 2.6, 3.4, 2.0, ["Evaluating", "Model"], "dia", fs=11)
    box(8.7, 3.3, 3.0, 1.0, ["AUC · F1 · Recall"], "metric", fs=10.5, bold_first=False)
    box(12.3, 3.3, 3.2, 1.2, ["Calibration", "Brier · ECE"], "metric", fs=10.5)
    box(8.7, 1.6, 3.0, 1.2, ["Recalibration +", "Decision Curve"], "metric", fs=10, bold_first=False)
    box(12.3, 1.6, 3.2, 1.2, ["SHAP +", "External Cohort"], "metric", fs=10.5)
    # Row 9: proposed model
    box(3.5, 0.65, 4.8, 1.0, ["Proposed Model:", "Confidence-Aware Self-Training"], "final", fs=11)

    # arrows
    arrow(cM, 18.4 - 0.85, 3.9, 16.0 + 0.75, rad=-0.12)   # data -> cleaning
    arrow(cM, 18.4 - 0.85, 7.5, 16.0 + 0.75, rad=0.12)    # data -> split
    arrow(3.9, 16.0 - 0.75, cM, 13.6 + 1.0, rad=0.10)     # cleaning -> diamond
    arrow(7.5, 16.0 - 0.75, cM, 13.6 + 1.0, rad=-0.10)    # split -> diamond
    arrow(cM, 13.6 - 1.0, 3.7, 11.4 + 0.7, rad=0.10)      # diamond -> scaler
    arrow(cM, 13.6 - 1.0, 7.7, 11.4 + 0.7, rad=-0.10)     # diamond -> raw
    arrow(3.7, 11.4 - 0.7, 3.6, cys + 0.55)               # scaler -> classifiers
    arrow(7.7, 11.4 - 0.7, 7.4, cys + 0.55)               # raw -> classifiers
    arrow(4.6, 8.85 - 0.2, 3.5, 6.9 + 0.85, rad=0.05)     # classifiers -> self-training
    arrow(6.6, 8.85 - 0.2, 10.0, 7.35 + 0.5, rad=-0.05)   # classifiers -> SSL
    arrow(3.5, 6.9 - 0.85, 4.6, 4.8 + 0.7, rad=0.12)      # self-training -> diagnostics
    arrow(10.2, 6.55 - 0.25, 7.2, 4.8 + 0.7, rad=-0.15)   # ssl -> diagnostics
    arrow(cM, 4.8 - 0.7, 3.5, 2.6 + 1.0, rad=0.10)        # diagnostics -> evaluating
    arrow(8.7 - 1.5, 3.3, 5.25, 2.72, rad=0.05)           # AUC -> evaluating
    arrow(12.3 - 1.6, 3.3 - 0.2, 5.3, 2.62, rad=-0.08)    # calibration -> evaluating
    arrow(8.7 - 1.5, 1.6, 5.25, 2.48, rad=-0.05)          # recalibration -> evaluating
    arrow(12.3 - 1.6, 1.6 + 0.2, 5.3, 2.4, rad=0.08)      # SHAP/external -> evaluating
    arrow(3.5, 2.6 - 1.0, 3.5, 0.65 + 0.5)                # evaluating -> proposed model

    plt.savefig(FIGURES_DIR / "workflow_diagram.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close()


def run_visualizations() -> pd.DataFrame:
    ensure_dirs()
    summary = build_summary_tables()
    plot_workflow_diagram()
    plot_dataset_distribution()
    plot_xgb_curves()
    plot_model_accuracy_variation()
    plot_supervised_vs_selftrained_accuracy()
    plot_threshold_comparison()
    plot_ssl_comparison()
    plot_error_propagation()
    plot_imbalance_analysis()
    plot_roc_curves("XGB", "roc_dynamic_xgb_by_ratio.pdf")
    plot_roc_curves("SVM", "roc_dynamic_svm_by_ratio.pdf")
    print("Saved visualizations and results/summary_table.csv", summary.shape)
    return summary


if __name__ == "__main__":
    run_visualizations()
