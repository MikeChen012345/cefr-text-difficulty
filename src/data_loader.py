"""Load and prepare the CEFR-SP dataset."""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset
from config import DATA_PATH, CEFR_TO_INT, SEED, N_FOLDS


from config import (
    DATASETS, DEFAULT_DATASET_KEY,
    dataset_csv_path, dataset_dir,
    CEFR_TO_INT, CEFR_LEVELS,
    SEED, N_FOLDS,
)

def _ensure_cached(dataset_key: str):
    """Download dataset from HF and cache splits as CSV if local file not found."""
    csv_train = dataset_csv_path(dataset_key, "train")
    if os.path.exists(csv_train):
        return

    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset_key={dataset_key}. Known: {list(DATASETS.keys())}")

    hf_name = DATASETS[dataset_key]
    ds = load_dataset(hf_name)

    out_dir = Path(dataset_dir(dataset_key))
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ds:
        df_split = ds[split_name].to_pandas()
        df_split.to_csv(dataset_csv_path(dataset_key, split_name), index=False)
        print(f"[cache] Saved {dataset_key}/{split_name}.csv: {len(df_split)} rows")

def load_data(dataset_key: str = DEFAULT_DATASET_KEY, split: str = "train") -> pd.DataFrame:
    """Load a UniversalCEFR dataset split and return DataFrame with numeric labels."""
    print("=" * 60)
    print(f"Loading dataset_key={dataset_key}, split={split}")

    _ensure_cached(dataset_key)

    path = dataset_csv_path(dataset_key, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cached split CSV: {path}")

    df = pd.read_csv(path)

    # Standardize schema: keep text + cefr_level if present
    # Most UniversalCEFR datasets provide these columns.
    keep_cols = [c for c in ["text", "cefr_level", "format", "source_name"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["text", "cefr_level"]).reset_index(drop=True)

    # Filter to strict CEFR levels only (avoid A1+, etc.)
    df = df[df["cefr_level"].isin(CEFR_LEVELS)].copy()

    df["label"] = df["cefr_level"].map(CEFR_TO_INT)
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    print(f"Loaded {len(df)} samples")
    print("Label distribution:\n", df["cefr_level"].value_counts().reindex(CEFR_LEVELS).fillna(0).astype(int))
    print("=" * 60)
    return df

def load_data_split(dataset_key: str = DEFAULT_DATASET_KEY, n_test_per_level: int = 50, split: str = "train"):
    """Balanced test set: sample n_test_per_level from each CEFR level, rest train."""
    df = load_data(dataset_key=dataset_key, split=split)

    rng = np.random.default_rng(SEED)
    test_indices = []

    for level in CEFR_LEVELS:
        level_idx = df.index[df["cefr_level"] == level].to_numpy()
        if len(level_idx) < n_test_per_level:
            raise ValueError(
                f"{dataset_key}: Not enough samples for {level}. "
                f"Have {len(level_idx)}, need {n_test_per_level}."
            )
        test_indices.extend(rng.choice(level_idx, size=n_test_per_level, replace=False))

    test_indices = np.array(test_indices)
    test_df = df.loc[test_indices].reset_index(drop=True)
    train_df = df.drop(test_indices).reset_index(drop=True)
    return train_df, test_df

def get_cv_splits(df: pd.DataFrame):
    """Return list of (train_idx, test_idx) for stratified k-fold CV."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(df, df["label"]))

if __name__ == "__main__":
    for key in DATASETS.keys():
        df = load_data(key)
        splits = get_cv_splits(df)
        print(f"{key}: {len(df)} rows, {len(splits)} folds")
