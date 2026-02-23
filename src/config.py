"""Configuration for the CEFR text difficulty experiments."""
import os

SEED = 42
N_FOLDS = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- Datasets ----------
# Keep CEFR-SP and add one more English corpus. Can add more later.
DATASETS = {
    "cefr_sp_en": "UniversalCEFR/cefr_sp_en",
    "readme_en": "UniversalCEFR/readme_en",
}

DEFAULT_DATASET_KEY = "cefr_sp_en"

# Local cache root for CSVs downloaded from HF
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets")

DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "cefr_sp_en", "train.csv")

def dataset_dir(dataset_key: str) -> str:
    return os.path.join(DATA_ROOT, dataset_key)

def dataset_csv_path(dataset_key: str, split: str = "train") -> str:
    return os.path.join(dataset_dir(dataset_key), f"{split}.csv")

# ---------- Output dirs (base) ----------
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
FIGURES_ROOT = os.path.join(PROJECT_ROOT, "figures")

def results_dir(dataset_key: str) -> str:
    return os.path.join(RESULTS_ROOT, dataset_key)

def figures_dir(dataset_key: str) -> str:
    return os.path.join(FIGURES_ROOT, dataset_key)

def models_dir(dataset_key: str) -> str:
    return os.path.join(results_dir(dataset_key), "models")

# ---------- Labels ----------
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_TO_INT = {level: i for i, level in enumerate(CEFR_LEVELS)}
INT_TO_CEFR = {i: level for i, level in enumerate(CEFR_LEVELS)}

# ---------- BERT training config ----------
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 64
BERT_LR = 2e-5
BERT_EPOCHS = 5
BERT_WARMUP_RATIO = 0.1

# GPU
DEVICE = "cuda:0"
