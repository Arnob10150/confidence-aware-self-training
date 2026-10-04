# Confidence-Aware Self-Training for CHD Risk Prediction

Code and experiments for the ICISET 2026 paper **"Confidence-Aware Self-Training for Coronary Heart Disease Risk Prediction Under Label Scarcity and Class Imbalance"** (Paper ID 205).

The paper asks whether confidence-based pseudo-labeling can be *trusted* under class imbalance, not merely whether it raises accuracy. It shows that a single global confidence threshold accepts pseudo-labels that are almost exclusively negative (0.1% positive against a true 15.2% prevalence), and fixes this with a class-aware adaptive threshold, with calibration corrected by default via Platt scaling.

## Key results

| Method (XGBoost, 40% labels, Framingham) | AUC | F1 | Recall | Precision |
|---|---|---|---|---|
| Standard threshold | 0.705 | 0.363 | 0.643 | 0.253 |
| Static-high threshold | 0.703 | 0.362 | 0.586 | 0.263 |
| Dynamic threshold | 0.695 | 0.341 | 0.690 | 0.227 |
| **Class-Aware ST (ours)** | 0.693 | 0.340 | **0.721** | 0.223 |
| CReST | 0.697 | 0.339 | 0.727 | 0.221 |
| Mean Teacher (surrogate) | 0.701 | 0.361 | 0.594 | 0.260 |
| Mean Teacher (deep MLP) | 0.707 | 0.359 | 0.634 | 0.253 |
| Label Propagation | 0.539 | 0.209 | 0.203 | 0.215 |

The recall gain over the standard threshold is significant (paired Wilcoxon, 5 seeds, p = 0.0625), and so is the precision cost (p = 0.0063); the AUC difference is **not** significant (p = 0.077). This is a recall-for-precision trade, not a free improvement.

Default Platt scaling reduces class-aware ECE from 0.364/0.372 to 0.017/0.024 at 5%/40% labels.

## Datasets

| Dataset | Source | Rows | Prevalence |
|---|---|---|---|
| Framingham Heart Study | Kaggle de-identified release | 4,240 | 15.2% |
| UCI Cleveland | UCI ML Repository, `processed.cleveland.data` | 303 | 45.9% |

`cross_check/uci_cleveland_processed.csv` is derived from the UCI file: the target is binarized (0 vs. 1-4) and the 6 missing values in `ca`/`thal` are median-imputed.

The Cox survival analysis in the paper additionally uses the NHLBI/BioLINCC Framingham teaching cohort (`frmgham2`), which carries day-level time-to-event fields. That file is **not redistributable** and is not included here; it must be requested from [BioLINCC](https://biolincc.nhlbi.nih.gov/teaching/).

## Setup

```bash
pip install -r requirements.txt
```

## Running

Scripts are numbered in dependency order and are run from the repository root.

```bash
python 1_preprocessing.py          # writes data/processed.csv
python run_all.py                  # steps 1-8: baselines, self-training, SSL, SHAP, figures
```

Then the extended analyses:

```bash
python 9_calibration_analysis.py     # Brier/ECE, reliability curves
python 11_recalibration_dca.py       # Platt/isotonic recalibration + decision-curve analysis
python 12_external_validation.py     # reduced-feature external check
python 13_deep_mean_teacher.py       # deep MLP Mean Teacher baseline
python 14_sensitivity_analysis.py    # threshold and hyperparameter sweeps
python 15_class_aware_crest.py       # class-aware adaptive threshold + CReST
python 16_significance_testing.py    # paired Wilcoxon / t-tests across seeds
python 17_mnar_simulation.py         # MNAR vs MCAR label hiding
python 18_class_aware_calibration.py # class-aware calibration + DCA
python 19_imbalance_fix_figure.py    # Fig. 5
python 20_reliability_figure.py      # Fig. 6
python 21_cleveland_multidataset.py  # Fig. 3, Cleveland CV ceiling
```

## Method

The class-aware threshold (Sec. III-C) follows FlexMatch: each class gets its own acceptance threshold, relaxed in proportion to how under-represented it is among the model's own predictions.

```
tau_t(c) = max(tau_min, tau_t * n_c / max_c' n_c')
```

where `n_c` counts unlabeled samples currently predicted (arg-max) as class `c`, `tau_t` is the dynamic schedule `max(0.95 - 0.05t, 0.70)`, and `tau_min = 0.65` floors the minority threshold short of accepting noise.

CReST accepts pseudo-labels at the same global dynamic threshold, then duplicates accepted pseudo-labels per class in inverse proportion to that class's labeled-pool share (capped at 1.6x) before retraining.

## Reproducibility

All stochastic steps (train/test split, model initialisation, self-training) use five fixed seeds: 42, 1, 2, 3, 4. Reported ± values are standard deviations across these seeds, and every significance test is paired across the same five matched seeds.

Classifier settings: LogisticRegression (`max_iter=1000`, `class_weight=balanced`), SVC (RBF, `probability=True`, `class_weight=balanced`), RandomForestClassifier (120 trees, `min_samples_leaf=2`, `class_weight=balanced`), XGBoost (120 estimators, `max_depth=3`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `scale_pos_weight = N_neg/N_pos`).

Figures are written with `pdf.fonttype = 42` (TrueType) rather than matplotlib's Type 3 default, which IEEE PDF eXpress rejects.

## Outputs

`results/` holds the generated CSVs; `results/figures/` holds the figures used in the paper. Notable files:

- `summary_table.csv` — mean/std aggregation across seeds for the paper tables
- `class_aware_crest_raw.csv` — per-seed class-aware and CReST results
- `significance_tests.csv` — all paired Wilcoxon and t-test results
- `mnar_simulation_raw.csv` — MCAR vs MNAR per-seed results
- `recalibration_metrics.csv` — raw/Platt/isotonic Brier and ECE
- `cleveland_cv_ceiling.csv` — full-label CV ceiling, both cohorts

## Citation

```bibtex
@inproceedings{anurag2026confidence,
  title     = {Confidence-Aware Self-Training for Coronary Heart Disease Risk
               Prediction Under Label Scarcity and Class Imbalance},
  author    = {Anurag, Arnob Aich and Islam, Shamiul and Risha, Nuren Tasnim
               and Hossain, Md. Sadman and Sultana, Faria},
  booktitle = {2026 5th International Conference on Innovations in Science,
               Engineering and Technology (ICISET)},
  year      = {2026},
  address   = {Chittagong, Bangladesh}
}
```

## Authors

Arnob Aich Anurag (corresponding), Shamiul Islam, Nuren Tasnim Risha, Md. Sadman Hossain, Faria Sultana — Department of Computer Science, American International University-Bangladesh (AIUB), Dhaka, Bangladesh.

## License

MIT License — see `LICENSE`.
