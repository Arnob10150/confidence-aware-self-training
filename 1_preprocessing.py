from __future__ import annotations

import pandas as pd

from utils import DATA_DIR, FRAMINGHAM_DIR, TARGET_COLUMN, ensure_dirs


RAW_PATH = FRAMINGHAM_DIR / "framingham_heart_study.csv"


def preprocess() -> pd.DataFrame:
    ensure_dirs()
    df = pd.read_csv(RAW_PATH)
    raw_shape = df.shape
    missing_before = df.isna().sum().to_dict()

    for column in df.columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna(df[column].median())

    assert not df.isnull().any().any(), "NaNs remain after preprocessing"
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df.to_csv(DATA_DIR / "processed.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "raw_rows": raw_shape[0],
                "raw_columns": raw_shape[1],
                "processed_rows": df.shape[0],
                "processed_columns": df.shape[1],
                "features": df.shape[1] - 1,
                "target_positive_rate": df[TARGET_COLUMN].mean(),
                "positive_count": int(df[TARGET_COLUMN].sum()),
                "negative_count": int((1 - df[TARGET_COLUMN]).sum()),
                "rows_with_missing_before": int(pd.read_csv(RAW_PATH).isna().any(axis=1).sum()),
            }
        ]
    )
    summary.to_csv("results/dataset_summary.csv", index=False)
    pd.Series(missing_before, name="missing_count").to_csv("results/missing_before.csv")
    print(summary.to_string(index=False))
    return df


if __name__ == "__main__":
    preprocess()
