from __future__ import annotations

"""Paired significance testing across matched seeds for every headline comparison
reviewers asked to see backed by statistics, rather than point estimates alone.

For each comparison we have 5 paired observations (seeds 42, 1, 2, 3, 4), the same
train/test split and labeled/unlabeled partition per seed across the methods being
compared. A paired Wilcoxon signed-rank test is used as the primary (distribution-free,
appropriate for n=5) test; a paired t-test is reported alongside for readers used to it.
With n=5 paired seeds, Wilcoxon's minimum two-sided p-value is 0.0625 (all 5 differences
same sign), so p<0.10 is the finest resolution available here and is reported as such
rather than implying false precision.
"""

import numpy as np
import pandas as pd
from scipy import stats

from utils import RESULTS_DIR


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = np.nan, 1.0
    t_stat, t_p = stats.ttest_rel(a, b)
    return {
        "n_pairs": len(a),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
        "wilcoxon_p": float(w_p),
        "ttest_stat": float(t_stat),
        "ttest_p": float(t_p),
    }


def compare_group(df: pd.DataFrame, group_col: str, metric: str, a_label: str, b_label: str, seed_col: str = "seed") -> dict:
    wide = df.pivot_table(index=seed_col, columns=group_col, values=metric)
    a = wide[a_label].dropna().values
    b = wide[b_label].reindex(wide[a_label].dropna().index).values
    return paired_test(a, b)


def run() -> pd.DataFrame:
    rows = []

    # 1) Self-training threshold variants vs supervised baseline, XGB @ 40% labels, Framingham.
    st = pd.read_csv(RESULTS_DIR / "selftraining_results.csv")
    base = pd.read_csv(RESULTS_DIR / "baseline_results.csv")
    xgb_st_40 = st[(st.model == "XGB") & (st.ratio == 0.40) & (st.config == "framingham")]
    xgb_base_40 = base[(base.model == "XGB") & (base.ratio == 0.40) & (base.config == "framingham")]
    for metric in ["auc", "recall", "f1", "precision"]:
        for variant in ["vanilla", "static_high", "dynamic"]:
            a = xgb_st_40[xgb_st_40.variant == variant].sort_values("seed")[metric].values
            b = xgb_base_40.sort_values("seed")[metric].values
            res = paired_test(a, b)
            rows.append({"comparison": f"{variant} ST vs supervised baseline (XGB, 40% labels)", "metric": metric, **res})

    # 2) Class-aware / CReST vs standard(vanilla)/dynamic threshold ST, XGB @ 40% labels.
    ca_crest = pd.read_csv(RESULTS_DIR / "class_aware_crest_raw.csv")
    ca_40 = ca_crest[(ca_crest.ratio == 0.40) & (ca_crest.config == "framingham")]
    for metric in ["auc", "recall", "f1", "precision"]:
        for method in ["ClassAwareST", "CReST"]:
            a = ca_40[ca_40.method == method].sort_values("seed")[metric].values
            for baseline_variant in ["vanilla", "dynamic"]:
                b = xgb_st_40[xgb_st_40.variant == baseline_variant].sort_values("seed")[metric].values
                res = paired_test(a, b)
                rows.append({
                    "comparison": f"{method} vs {baseline_variant} threshold ST (XGB, 40% labels)",
                    "metric": metric, **res,
                })
        # Class-aware vs CReST directly.
        a = ca_40[ca_40.method == "ClassAwareST"].sort_values("seed")["recall"].values
        b = ca_40[ca_40.method == "CReST"].sort_values("seed")["recall"].values
        rows.append({"comparison": "ClassAwareST vs CReST (XGB, 40% labels)", "metric": "recall", **paired_test(a, b)})
        a_auc = ca_40[ca_40.method == "ClassAwareST"].sort_values("seed")["auc"].values
        b_auc = ca_40[ca_40.method == "CReST"].sort_values("seed")["auc"].values
        rows.append({"comparison": "ClassAwareST vs CReST (XGB, 40% labels)", "metric": "auc", **paired_test(a_auc, b_auc)})

    # 3) SSL baselines (Mean Teacher, Label Propagation) vs class-aware, XGB @ 40% labels.
    ssl = pd.read_csv(RESULTS_DIR / "ssl_comparison.csv")
    ssl_40 = ssl[(ssl.ratio == 0.40) & (ssl.config == "framingham")]
    for metric in ["auc", "recall"]:
        ca_vals = ca_40[ca_40.method == "ClassAwareST"].sort_values("seed")[metric].values
        for method in ["MeanTeacher", "LabelPropagation"]:
            b = ssl_40[ssl_40.method == method].sort_values("seed")[metric].values
            rows.append({
                "comparison": f"ClassAwareST vs {method} (XGB, 40% labels)",
                "metric": metric, **paired_test(ca_vals, b),
            })

    # 4) Calibration: raw vs Platt-recalibrated ECE, supervised/dynamic/class-aware XGB.
    recal = pd.read_csv(RESULTS_DIR / "recalibration_metrics.csv")
    for method in recal["method"].unique():
        for ratio in [0.05, 0.40]:
            sub = recal[(recal.method == method) & (recal.ratio == ratio)].sort_values("seed")
            if len(sub) < 2:
                continue
            res = paired_test(sub["raw_ece"].values, sub["platt_ece"].values)
            rows.append({"comparison": f"{method} raw vs Platt ECE (ratio={ratio})", "metric": "ece", **res})

    # 5) MNAR vs MCAR (random) label hiding -- added once 17_mnar_simulation.py has run.
    mnar_path = RESULTS_DIR / "mnar_simulation_raw.csv"
    if mnar_path.exists():
        mnar = pd.read_csv(mnar_path)
        for method in mnar["method"].unique():
            for ratio in sorted(mnar["ratio"].unique()):
                for metric in ["auc", "recall"]:
                    sub = mnar[(mnar.method == method) & (mnar.ratio == ratio)]
                    wide = sub.pivot_table(index="seed", columns="scenario", values=metric)
                    if "MCAR" in wide.columns and "MNAR" in wide.columns:
                        a = wide["MCAR"].dropna().values
                        b = wide["MNAR"].reindex(wide["MCAR"].dropna().index).values
                        rows.append({
                            "comparison": f"{method} MCAR vs MNAR (ratio={ratio})",
                            "metric": metric, **paired_test(a, b),
                        })

    out = pd.DataFrame(rows)
    out["significant_p05"] = out["wilcoxon_p"] < 0.05
    out["significant_p10"] = out["wilcoxon_p"] < 0.10
    out.to_csv(RESULTS_DIR / "significance_tests.csv", index=False)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    run()
