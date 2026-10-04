from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils import RATIOS, RESULTS_DIR, SEEDS, ensure_dirs, get_splits, load_processed

PRIMARY = "framingham"
HIDDEN = 32
NOISE_STD = 0.10
EMA_DECAY = 0.99
EPOCHS = 120
RAMPUP_EPOCHS = 40
MAX_CONSISTENCY_WEIGHT = 1.0
LR = 1e-3


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def train_deep_mean_teacher(X_l, y_l, X_u, X_test, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler().fit(X_l)
    Xl = torch.tensor(scaler.transform(X_l), dtype=torch.float32)
    yl = torch.tensor(y_l, dtype=torch.float32)
    Xu = torch.tensor(scaler.transform(X_u), dtype=torch.float32) if len(X_u) > 0 else None
    Xt = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

    n_features = Xl.shape[1]
    student = MLP(n_features)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)

    pos_weight = torch.tensor((y_l == 0).sum() / max((y_l == 1).sum(), 1), dtype=torch.float32)
    sup_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        student.train()
        optimizer.zero_grad()

        sup_logits = student(Xl + NOISE_STD * torch.randn_like(Xl))
        loss = sup_criterion(sup_logits, yl)

        if Xu is not None and len(Xu) > 0:
            ramp = min(1.0, epoch / RAMPUP_EPOCHS)
            consistency_weight = MAX_CONSISTENCY_WEIGHT * ramp
            student_logits_u = student(Xu + NOISE_STD * torch.randn_like(Xu))
            with torch.no_grad():
                teacher_logits_u = teacher(Xu)
            consistency_loss = torch.mean((torch.sigmoid(student_logits_u) - torch.sigmoid(teacher_logits_u)) ** 2)
            loss = loss + consistency_weight * consistency_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.mul_(EMA_DECAY).add_(s_param, alpha=1 - EMA_DECAY)

    teacher.eval()
    with torch.no_grad():
        prob = torch.sigmoid(teacher(Xt)).numpy()
    return prob


def run(cal_ratios: list[float] = (0.05, 0.40)) -> pd.DataFrame:
    ensure_dirs()
    X, y = load_processed(PRIMARY)
    rows = []
    for ratio in cal_ratios:
        for seed in SEEDS:
            X_l, y_l, X_u, y_u, X_test, y_test = get_splits(X, y, ratio, seed)
            prob = train_deep_mean_teacher(X_l, y_l, X_u, X_test, seed)
            pred = (prob >= 0.5).astype(int)
            rows.append(
                {
                    "ratio": ratio,
                    "seed": seed,
                    "method": "Deep Mean Teacher (MLP)",
                    "auc": roc_auc_score(y_test, prob),
                    "f1": f1_score(y_test, pred, zero_division=0),
                    "recall": recall_score(y_test, pred, zero_division=0),
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "brier": brier_score(np.asarray(y_test), prob),
                    "ece": expected_calibration_error(np.asarray(y_test), prob),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "deep_mean_teacher_raw.csv", index=False)
    summary = df.groupby("ratio").mean(numeric_only=True).drop(columns="seed").reset_index()
    summary.to_csv(RESULTS_DIR / "deep_mean_teacher_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
