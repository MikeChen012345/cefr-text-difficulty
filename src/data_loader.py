"""Load and prepare the CEFR-SP dataset."""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset
from config import DATA_PATH, CEFR_TO_INT, SEED, N_FOLDS


def load_data():
    """Load CEFR-SP dataset and return DataFrame with numeric labels."""
    if not os.path.exists(DATA_PATH): # download from Hugging Face if local file not found
        dataset = load_dataset("UniversalCEFR/cefr_sp_en")
        # Save each split to CSV
        for split_name in dataset:
            df = dataset[split_name].to_pandas()
            df.to_csv(f"datasets/cefr_sp_en/{split_name}.csv", index=False)
            print(f"Saved {split_name}: {len(df)} rows")

    df = pd.read_csv(DATA_PATH)
    df = df[["text", "cefr_level"]].dropna().reset_index(drop=True)
    df["label"] = df["cefr_level"].map(CEFR_TO_INT)
    print(f"Loaded {len(df)} sentences")
    print(f"Label distribution:\n{df['cefr_level'].value_counts().sort_index()}")
    return df


def load_data_split():
    """Load Universal CEFR dataset from Hugging Face and do a train/test split.
    Choose 50 random samples from each CEFR level for the test set, and use the rest for training. Return train and test DataFrames with numeric labels.
    
    Returns:
        train_df: DataFrame containing training samples with numeric labels.
        test_df: DataFrame containing test samples with numeric labels.
    """
    df = load_data()
    np.random.seed(SEED)
    test_indices = []
    for level in df["cefr_level"].unique():
        level_indices = df[df["cefr_level"] == level].index
        test_indices.extend(np.random.choice(level_indices, size=50, replace=False))
    test_df = df.loc[test_indices].reset_index(drop=True)
    train_df = df.drop(test_indices).reset_index(drop=True)
    return train_df, test_df


def get_cv_splits(df):
    """Return list of (train_idx, test_idx) for stratified k-fold CV."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(df, df["label"]))



if __name__ == "__main__":
    df = load_data()
    splits = get_cv_splits(df)
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {i}: train={len(train_idx)}, test={len(test_idx)}")
