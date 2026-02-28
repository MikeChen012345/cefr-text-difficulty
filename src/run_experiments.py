"""Main experiment runner for CEFR text difficulty modeling.

Runs all experiments:
1. Feature extraction and EDA
2. Interpretable classifier training with ablation
3. Neural model training (BERT + DAN)
4. Diagnostic experiments (probing, error analysis)
"""
import os
import sys
import json
import time
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, cohen_kappa_score, r2_score
)
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

from config import *
from config import results_dir, figures_dir
from data_loader import load_data, get_cv_splits

# ── Reproducibility ───────────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

def log(msg=""):
    print(msg)
    sys.stdout.flush()

log(f"Python: {sys.version}")
log(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: EDA and Feature Analysis
# ══════════════════════════════════════════════════════════════════════

def run_eda(df, features_df, feature_groups, dataset_key: str):
    log("\n" + "="*60)
    log("EXPERIMENT 1: EDA & Feature Analysis")
    log("="*60)

    # Class distribution plot
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["cefr_level"].value_counts().reindex(CEFR_LEVELS)
    counts.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", 6))
    ax.set_xlabel("CEFR Level")
    ax.set_ylabel("Count")
    ax.set_title(f"CEFR Level Distribution ({dataset_key})")
    ax.set_xticklabels(CEFR_LEVELS, rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 30, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    # Feature correlations with ordinal CEFR level
    labels = df["label"].values
    correlations = {}
    for col in features_df.columns:
        rho, p = stats.spearmanr(features_df[col].values, labels)
        correlations[col] = {"rho": rho, "p": p}

    corr_df = pd.DataFrame(correlations).T
    corr_df["abs_rho"] = corr_df["rho"].abs()
    corr_df = corr_df.sort_values("abs_rho", ascending=False)

    # Top 20 features by correlation
    top20 = corr_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in top20["rho"]]
    ax.barh(range(len(top20)), top20["rho"], color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=9)
    ax.set_xlabel("Spearman rho with CEFR Level (ordinal)")
    ax.set_title("Top 20 Features Correlated with CEFR Difficulty")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_correlations_top20.png"), dpi=150)
    plt.close()

    # Group-level correlation summary
    group_corrs = {}
    for group, cols in feature_groups.items():
        valid_cols = [c for c in cols if c in corr_df.index]
        if valid_cols:
            group_corrs[group] = {
                "mean_abs_rho": float(corr_df.loc[valid_cols, "abs_rho"].mean()),
                "max_abs_rho": float(corr_df.loc[valid_cols, "abs_rho"].max()),
                "best_feature": corr_df.loc[valid_cols, "abs_rho"].idxmax(),
                "n_features": len(valid_cols),
            }
    group_corr_df = pd.DataFrame(group_corrs).T.sort_values("mean_abs_rho", ascending=False)
    log("\nFeature Group Correlation Summary:")
    log(group_corr_df.to_string())

    # Feature distributions by CEFR level (for top features)
    top5_features = corr_df.head(5).index.tolist()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, feat in zip(axes, top5_features):
        level_data = []
        level_labels = []
        for i in range(6):
            d = features_df.loc[df["label"] == i, feat].values
            if len(d) > 0:
                level_data.append(d)
                level_labels.append(CEFR_LEVELS[i])
        ax.boxplot(level_data, labels=level_labels)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("CEFR Level")
    plt.suptitle("Top 5 Features by CEFR Level", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "top_features_by_level.png"), dpi=150)
    plt.close()

    corr_df.to_csv(os.path.join(RESULTS_DIR, "feature_correlations.csv"))
    return corr_df, group_corr_df


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Interpretable Classifiers with Ablation
# ══════════════════════════════════════════════════════════════════════

def eval_classifier(model_fn, X, y, splits, scale=False):
    """Evaluate a classifier with cross-validation. Returns fold results."""
    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_results.append({
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "accuracy": accuracy_score(y_test, y_pred),
            "per_class_f1": f1_score(y_test, y_pred, average=None, labels=list(range(6))),
            "y_pred": y_pred,
            "y_test": y_test,
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(range(6))),
            "adjacent_acc": float(np.mean(np.abs(y_test - y_pred) <= 1)),
            "kappa": cohen_kappa_score(y_test, y_pred, weights="quadratic"),
        })
    return fold_results


def summarize_folds(fold_results):
    """Compute summary stats from fold results."""
    mean_f1 = np.mean([r["macro_f1"] for r in fold_results])
    std_f1 = np.std([r["macro_f1"] for r in fold_results])
    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    std_acc = np.std([r["accuracy"] for r in fold_results])
    mean_adj = np.mean([r["adjacent_acc"] for r in fold_results])
    mean_kappa = np.mean([r["kappa"] for r in fold_results])
    return {
        "macro_f1": f"{mean_f1:.4f} +/- {std_f1:.4f}",
        "accuracy": f"{mean_acc:.4f} +/- {std_acc:.4f}",
        "adjacent_acc": f"{mean_adj:.4f}",
        "kappa": f"{mean_kappa:.4f}",
        "macro_f1_mean": mean_f1,
        "macro_f1_std": std_f1,
        "accuracy_mean": mean_acc,
        "fold_results": fold_results,
    }


def run_interpretable_models(features_df, labels, splits, feature_groups):
    log("\n" + "="*60)
    log("EXPERIMENT 2: Interpretable Classifiers")
    log("="*60)

    X = features_df.values
    y = labels.values
    feature_names = features_df.columns.tolist()

    # Use GPU-accelerated histogram method for XGBoost
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

    models = {
        "LogReg": (
            lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=SEED, class_weight="balanced"),
            True  # scale
        ),
        "RandomForest": (
            lambda: RandomForestClassifier(n_estimators=100, max_depth=15, random_state=SEED, class_weight="balanced"),
            False
        ),
        "XGBoost": (
            lambda: xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=SEED, eval_metric="mlogloss",
                tree_method="hist", device=xgb_device,
            ),
            False
        ),
    }

    all_results = {}

    for model_name, (model_fn, scale) in models.items():
        t0 = time.time()
        fold_results = eval_classifier(model_fn, X, y, splits, scale=scale)
        summary = summarize_folds(fold_results)
        all_results[f"Full_{model_name}"] = summary
        elapsed = time.time() - t0
        log(f"{model_name} (all features): F1={summary['macro_f1_mean']:.4f}+/-{summary['macro_f1_std']:.4f}, "
            f"Acc={summary['accuracy_mean']:.4f} [{elapsed:.1f}s]")

    # Majority baseline
    majority_f1s = []
    for train_idx, test_idx in splits:
        y_train, y_test = y[train_idx], y[test_idx]
        majority_class = np.bincount(y_train).argmax()
        y_pred = np.full_like(y_test, majority_class)
        majority_f1s.append(f1_score(y_test, y_pred, average="macro"))
    all_results["Majority"] = {"macro_f1_mean": np.mean(majority_f1s)}
    log(f"Majority baseline: F1={np.mean(majority_f1s):.4f}")

    # Feature group ablation (using XGBoost as best model)
    log("\n--- Feature Group Ablation (XGBoost) ---")
    ablation_results = {}

    xgb_fn = lambda: xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=SEED, eval_metric="mlogloss",
        tree_method="hist", device=xgb_device,
    )

    for group_name, group_cols in feature_groups.items():
        col_idxs = [feature_names.index(c) for c in group_cols if c in feature_names]
        if not col_idxs:
            continue

        X_sub = X[:, col_idxs]
        fold_results = eval_classifier(xgb_fn, X_sub, y, splits)
        mean_f1 = np.mean([r["macro_f1"] for r in fold_results])
        std_f1 = np.std([r["macro_f1"] for r in fold_results])
        ablation_results[f"Only_{group_name}"] = {
            "macro_f1_mean": mean_f1, "macro_f1_std": std_f1,
            "n_features": len(col_idxs),
        }
        log(f"  {group_name} only ({len(col_idxs)} feats): F1={mean_f1:.4f}+/-{std_f1:.4f}")

    # Leave-one-group-out
    for group_name, group_cols in feature_groups.items():
        exclude_idxs = set(feature_names.index(c) for c in group_cols if c in feature_names)
        keep_idxs = [i for i in range(X.shape[1]) if i not in exclude_idxs]
        if not keep_idxs:
            continue

        X_sub = X[:, keep_idxs]
        fold_results = eval_classifier(xgb_fn, X_sub, y, splits)
        mean_f1 = np.mean([r["macro_f1"] for r in fold_results])
        full_f1 = all_results["Full_XGBoost"]["macro_f1_mean"]
        drop = full_f1 - mean_f1
        ablation_results[f"Without_{group_name}"] = {
            "macro_f1_mean": mean_f1, "drop_from_full": drop,
        }
        log(f"  Without {group_name}: F1={mean_f1:.4f} (drop={drop:+.4f})")

    all_results["ablation"] = ablation_results

    # Feature importance (XGBoost on full data)
    log("\n--- Feature Importance ---")
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=SEED, eval_metric="mlogloss",
        tree_method="hist", device=xgb_device,
    )
    model.fit(X, y)
    importances = model.feature_importances_

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    top20 = imp_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top20)), top20["importance"].values, color=sns.color_palette("viridis", 20))
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values, fontsize=9)
    ax.set_xlabel("XGBoost Feature Importance (gain)")
    ax.set_title("Top 20 Features by XGBoost Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "xgboost_feature_importance.png"), dpi=150)
    plt.close()
    imp_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)

    log("Feature models complete.")
    return all_results, ablation_results, imp_df


def run_tfidf_linear_svm(df, splits):
    log("\n" + "="*60)
    log("EXPERIMENT 2b: TF-IDF + Linear SVM")
    log("="*60)

    texts = df["text"].tolist()
    y = df["label"].values
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        t0 = time.time()
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, TFIDF_NGRAM_MAX),
            min_df=TFIDF_MIN_DF,
            max_features=TFIDF_MAX_FEATURES,
            sublinear_tf=True,
        )
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        model = LinearSVC(C=SVM_C, class_weight="balanced", random_state=SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_results.append({
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "accuracy": accuracy_score(y_test, y_pred),
            "per_class_f1": f1_score(y_test, y_pred, average=None, labels=list(range(6))),
            "y_pred": y_pred,
            "y_true": y_test,
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(range(6))),
            "adjacent_acc": float(np.mean(np.abs(y_test - y_pred) <= 1)),
            "kappa": cohen_kappa_score(y_test, y_pred, weights="quadratic"),
            "vocab_size": int(X_train.shape[1]),
        })
        log(f"  Fold {fold_i+1}/{N_FOLDS}: F1={fold_results[-1]['macro_f1']:.4f}, "
            f"Acc={fold_results[-1]['accuracy']:.4f}, Vocab={X_train.shape[1]} "
            f"[{time.time()-t0:.1f}s]")

    mean_f1 = np.mean([r["macro_f1"] for r in fold_results])
    std_f1 = np.std([r["macro_f1"] for r in fold_results])
    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    std_acc = np.std([r["accuracy"] for r in fold_results])
    mean_adj = np.mean([r["adjacent_acc"] for r in fold_results])
    mean_kappa = np.mean([r["kappa"] for r in fold_results])
    mean_vocab = np.mean([r["vocab_size"] for r in fold_results])

    results = {
        "macro_f1": f"{mean_f1:.4f} +/- {std_f1:.4f}",
        "accuracy": f"{mean_acc:.4f} +/- {std_acc:.4f}",
        "adjacent_acc": f"{mean_adj:.4f}",
        "kappa": f"{mean_kappa:.4f}",
        "macro_f1_mean": mean_f1,
        "macro_f1_std": std_f1,
        "accuracy_mean": mean_acc,
        "avg_vocab_size": float(mean_vocab),
        "fold_results": fold_results,
    }

    per_class_f1 = np.mean([r["per_class_f1"] for r in fold_results], axis=0)
    results["per_class_f1_avg"] = {CEFR_LEVELS[i]: float(per_class_f1[i]) for i in range(6)}

    log(f"TF-IDF + LinearSVM Final: F1={mean_f1:.4f}+/-{std_f1:.4f}, "
        f"Acc={mean_acc:.4f}+/-{std_acc:.4f}")
    return results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: BERT Fine-tuning
# ══════════════════════════════════════════════════════════════════════

class CEFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=BERT_MAX_LEN):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                    max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class DANTextDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


class DANClassifier(nn.Module):
    def __init__(self, vocab_size, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, DAN_EMBED_DIM, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(DAN_EMBED_DIM, DAN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DAN_DROPOUT),
            nn.Linear(DAN_HIDDEN_DIM, 6),
        )

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)  # [B, T, D]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        avg = summed / counts
        return self.classifier(avg)


def encode_texts_with_tokenizer(texts, tokenizer, max_len=DAN_MAX_LEN):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return encodings["input_ids"], encodings["attention_mask"]


def train_bert_fold(texts, labels, train_idx, test_idx, fold_i):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    train_labels = labels[train_idx].tolist()
    test_labels = labels[test_idx].tolist()

    train_ds = CEFRDataset(train_texts, train_labels, tokenizer)
    test_ds = CEFRDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BERT_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BERT_BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=6
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)
    total_steps = len(train_loader) * BERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * BERT_WARMUP_RATIO),
        num_training_steps=total_steps
    )

    best_f1 = 0
    best_preds = None
    best_logits_all = None

    for epoch in range(BERT_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        model.eval()
        all_preds = []
        all_logits = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        all_preds = np.array(all_preds)
        all_logits = np.vstack(all_logits)
        f1 = f1_score(test_labels, all_preds, average="macro")
        acc = accuracy_score(test_labels, all_preds)
        log(f"  Fold {fold_i} Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, F1={f1:.4f}, Acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_preds = all_preds.copy()
            best_logits_all = all_logits.copy()

    y_test = np.array(test_labels)
    del model
    torch.cuda.empty_cache()

    return {
        "macro_f1": best_f1,
        "accuracy": accuracy_score(y_test, best_preds),
        "per_class_f1": f1_score(y_test, best_preds, average=None, labels=list(range(6))),
        "y_pred": best_preds,
        "y_true": y_test,
        "logits": best_logits_all,
        "confusion_matrix": confusion_matrix(y_test, best_preds, labels=list(range(6))),
        "adjacent_acc": float(np.mean(np.abs(y_test - best_preds) <= 1)),
        "kappa": cohen_kappa_score(y_test, best_preds, weights="quadratic"),
    }


def run_bert(df, splits):
    log("\n" + "="*60)
    log("EXPERIMENT 3a: BERT Fine-tuning")
    log("="*60)

    texts = df["text"].tolist()
    labels = df["label"].values

    fold_results = []
    all_logits = {}

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        log(f"\n--- Fold {fold_i+1}/{N_FOLDS} ---")
        t0 = time.time()
        result = train_bert_fold(texts, labels, train_idx, test_idx, fold_i)
        fold_results.append(result)
        for i, idx in enumerate(test_idx):
            all_logits[idx] = result["logits"][i]
        log(f"  Fold {fold_i+1} done: F1={result['macro_f1']:.4f} [{time.time()-t0:.0f}s]")

    mean_f1 = np.mean([r["macro_f1"] for r in fold_results])
    std_f1 = np.std([r["macro_f1"] for r in fold_results])
    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    std_acc = np.std([r["accuracy"] for r in fold_results])
    mean_adj = np.mean([r["adjacent_acc"] for r in fold_results])
    mean_kappa = np.mean([r["kappa"] for r in fold_results])

    log(f"\nBERT Final: F1={mean_f1:.4f}+/-{std_f1:.4f}, Acc={mean_acc:.4f}+/-{std_acc:.4f}")

    bert_results = {
        "macro_f1": f"{mean_f1:.4f} +/- {std_f1:.4f}",
        "accuracy": f"{mean_acc:.4f} +/- {std_acc:.4f}",
        "adjacent_acc": f"{mean_adj:.4f}",
        "kappa": f"{mean_kappa:.4f}",
        "macro_f1_mean": mean_f1,
        "macro_f1_std": std_f1,
        "accuracy_mean": mean_acc,
        "fold_results": fold_results,
    }

    per_class_f1 = np.mean([r["per_class_f1"] for r in fold_results], axis=0)
    bert_results["per_class_f1_avg"] = {CEFR_LEVELS[i]: float(per_class_f1[i]) for i in range(6)}

    logits_array = np.zeros((len(df), 6))
    for idx, logit in all_logits.items():
        logits_array[idx] = logit

    return bert_results, logits_array


def train_dan_fold(texts, labels, train_idx, test_idx, fold_i, tokenizer, pretrained_embeddings=None):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    train_labels = labels[train_idx].tolist()
    test_labels = labels[test_idx].tolist()

    train_ids, train_masks = encode_texts_with_tokenizer(train_texts, tokenizer)
    test_ids, test_masks = encode_texts_with_tokenizer(test_texts, tokenizer)

    train_ds = DANTextDataset(train_ids, train_masks, train_labels)
    test_ds = DANTextDataset(test_ids, test_masks, test_labels)

    train_loader = DataLoader(train_ds, batch_size=DAN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=DAN_BATCH_SIZE)

    model = DANClassifier(tokenizer.vocab_size, pretrained_embeddings=pretrained_embeddings).to(device)

    if DAN_USE_CLASS_WEIGHTS:
        class_counts = np.bincount(np.array(train_labels), minlength=6)
        class_weights = len(train_labels) / (6 * np.maximum(class_counts, 1))
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=DAN_LR, weight_decay=DAN_WEIGHT_DECAY)

    best_f1 = 0.0
    best_preds = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(DAN_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_batch = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())

        all_preds = np.array(all_preds)
        f1 = f1_score(test_labels, all_preds, average="macro")
        acc = accuracy_score(test_labels, all_preds)
        log(f"  Fold {fold_i} Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, F1={f1:.4f}, Acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_preds = all_preds.copy()
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= DAN_PATIENCE:
            log(f"  Fold {fold_i} early stop at epoch {epoch+1} (best epoch={best_epoch}, best F1={best_f1:.4f})")
            break

    y_test = np.array(test_labels)
    del model
    torch.cuda.empty_cache()

    return {
        "macro_f1": best_f1,
        "accuracy": accuracy_score(y_test, best_preds),
        "per_class_f1": f1_score(y_test, best_preds, average=None, labels=list(range(6))),
        "y_pred": best_preds,
        "y_true": y_test,
        "confusion_matrix": confusion_matrix(y_test, best_preds, labels=list(range(6))),
        "adjacent_acc": float(np.mean(np.abs(y_test - best_preds) <= 1)),
        "kappa": cohen_kappa_score(y_test, best_preds, weights="quadratic"),
        "vocab_size": tokenizer.vocab_size,
        "best_epoch": best_epoch,
    }


def run_dan(df, splits):
    log("\n" + "="*60)
    log("EXPERIMENT 3b: DAN")
    log("="*60)

    texts = df["text"].tolist()
    labels = df["label"].values
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    pretrained_embeddings = None
    if DAN_INIT_FROM_BERT:
        log("Initializing DAN embeddings from bert-base-uncased...")
        base_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=6)
        pretrained_embeddings = base_model.bert.embeddings.word_embeddings.weight.detach().clone()
        del base_model

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        log(f"\n--- Fold {fold_i+1}/{N_FOLDS} ---")
        t0 = time.time()
        result = train_dan_fold(
            texts, labels, train_idx, test_idx, fold_i,
            tokenizer=tokenizer,
            pretrained_embeddings=pretrained_embeddings,
        )
        fold_results.append(result)
        log(f"  Fold {fold_i+1} done: F1={result['macro_f1']:.4f} [{time.time()-t0:.0f}s]")

    mean_f1 = np.mean([r["macro_f1"] for r in fold_results])
    std_f1 = np.std([r["macro_f1"] for r in fold_results])
    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    std_acc = np.std([r["accuracy"] for r in fold_results])
    mean_adj = np.mean([r["adjacent_acc"] for r in fold_results])
    mean_kappa = np.mean([r["kappa"] for r in fold_results])
    mean_vocab = np.mean([r["vocab_size"] for r in fold_results])

    log(f"\nDAN Final: F1={mean_f1:.4f}+/-{std_f1:.4f}, Acc={mean_acc:.4f}+/-{std_acc:.4f}")

    dan_results = {
        "macro_f1": f"{mean_f1:.4f} +/- {std_f1:.4f}",
        "accuracy": f"{mean_acc:.4f} +/- {std_acc:.4f}",
        "adjacent_acc": f"{mean_adj:.4f}",
        "kappa": f"{mean_kappa:.4f}",
        "macro_f1_mean": mean_f1,
        "macro_f1_std": std_f1,
        "accuracy_mean": mean_acc,
        "avg_vocab_size": float(mean_vocab),
        "fold_results": fold_results,
    }

    per_class_f1 = np.mean([r["per_class_f1"] for r in fold_results], axis=0)
    dan_results["per_class_f1_avg"] = {CEFR_LEVELS[i]: float(per_class_f1[i]) for i in range(6)}

    return dan_results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Diagnostic Experiments
# ══════════════════════════════════════════════════════════════════════

def run_diagnostics(features_df, labels, bert_logits, splits, feature_groups,
                    feature_results, bert_results):
    log("\n" + "="*60)
    log("EXPERIMENT 4: Diagnostic Experiments")
    log("="*60)

    X = features_df.values
    y = labels.values
    feature_names = features_df.columns.tolist()

    # 4a: Feature Regression on BERT Predictions
    log("\n--- 4a: Feature Regression on BERT Predictions ---")
    bert_preds_all = bert_logits.argmax(axis=1)

    probing_r2 = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        bert_pred_train = bert_preds_all[train_idx]
        bert_pred_test = bert_preds_all[test_idx]

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, bert_pred_train)
        pred = ridge.predict(X_test)
        r2 = r2_score(bert_pred_test, pred)
        probing_r2.append(r2)

    mean_r2 = np.mean(probing_r2)
    std_r2 = np.std(probing_r2)
    log(f"  Mean R2 = {mean_r2:.4f} +/- {std_r2:.4f}")

    # 4b: Which features explain BERT?
    log("\n--- 4b: Top features explaining BERT predictions ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(X_scaled, bert_preds_all)
    coef_importance = np.abs(ridge_full.coef_)
    coef_df = pd.DataFrame({"feature": feature_names, "ridge_coef_abs": coef_importance})
    coef_df = coef_df.sort_values("ridge_coef_abs", ascending=False)
    for _, row in coef_df.head(10).iterrows():
        log(f"    {row['feature']}: {row['ridge_coef_abs']:.4f}")

    # Group-level probing R2
    log("\n--- 4b2: Group-level probing R2 ---")
    group_r2 = {}
    for group_name, group_cols in feature_groups.items():
        col_idxs = [feature_names.index(c) for c in group_cols if c in feature_names]
        if not col_idxs:
            continue
        X_group = X[:, col_idxs]
        fold_r2s = []
        for train_idx, test_idx in splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_group[train_idx])
            X_te = scaler.transform(X_group[test_idx])
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tr, bert_preds_all[train_idx])
            pred = ridge.predict(X_te)
            fold_r2s.append(r2_score(bert_preds_all[test_idx], pred))
        group_r2[group_name] = np.mean(fold_r2s)
        log(f"  {group_name}: R2={np.mean(fold_r2s):.4f}")

    # 4c: Error analysis
    log("\n--- 4c: Error Analysis ---")
    xgb_preds_all = np.zeros(len(y), dtype=int)
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        fold_res = feature_results["Full_XGBoost"]["fold_results"][fold_i]
        xgb_preds_all[test_idx] = fold_res["y_pred"]

    both_correct = (xgb_preds_all == y) & (bert_preds_all == y)
    xgb_only = (xgb_preds_all == y) & (bert_preds_all != y)
    bert_only = (xgb_preds_all != y) & (bert_preds_all == y)
    both_wrong = (xgb_preds_all != y) & (bert_preds_all != y)

    log(f"  Both correct: {both_correct.sum()} ({100*both_correct.mean():.1f}%)")
    log(f"  XGBoost only correct: {xgb_only.sum()} ({100*xgb_only.mean():.1f}%)")
    log(f"  BERT only correct: {bert_only.sum()} ({100*bert_only.mean():.1f}%)")
    log(f"  Both wrong: {both_wrong.sum()} ({100*both_wrong.mean():.1f}%)")

    # 4d: Confusion matrices
    xgb_cm = np.zeros((6, 6), dtype=int)
    bert_cm = np.zeros((6, 6), dtype=int)
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        xgb_cm += feature_results["Full_XGBoost"]["fold_results"][fold_i]["confusion_matrix"]
        bert_cm += bert_results["fold_results"][fold_i]["confusion_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, cm, title in zip(axes, [xgb_cm, bert_cm], ["XGBoost (Features)", "BERT"]):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CEFR_LEVELS, yticklabels=CEFR_LEVELS, ax=ax,
                    vmin=0, vmax=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix: {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices_comparison.png"), dpi=150)
    plt.close()

    # 4e: Per-class F1 comparison
    xgb_per_class = np.mean([r["per_class_f1"] for r in feature_results["Full_XGBoost"]["fold_results"]], axis=0)
    bert_per_class = np.mean([r["per_class_f1"] for r in bert_results["fold_results"]], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(6)
    width = 0.35
    ax.bar(x - width/2, xgb_per_class, width, label="XGBoost (Features)", color="#3498db")
    ax.bar(x + width/2, bert_per_class, width, label="BERT", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(CEFR_LEVELS)
    ax.set_xlabel("CEFR Level")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1: Features vs. BERT")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "per_class_f1_comparison.png"), dpi=150)
    plt.close()

    # 4f: BERT advantage by level
    bert_advantage_by_level = {}
    for level in range(6):
        mask = y == level
        if mask.sum() == 0:
            continue
        xgb_acc = float((xgb_preds_all[mask] == y[mask]).mean())
        bert_acc = float((bert_preds_all[mask] == y[mask]).mean())
        bert_advantage_by_level[CEFR_LEVELS[level]] = {
            "xgb_acc": xgb_acc,
            "bert_acc": bert_acc,
            "bert_advantage": bert_acc - xgb_acc,
            "n_samples": int(mask.sum()),
        }
    log("\n  BERT advantage by level:")
    for level, data in bert_advantage_by_level.items():
        log(f"    {level}: XGB={data['xgb_acc']:.3f}, BERT={data['bert_acc']:.3f}, "
            f"delta={data['bert_advantage']:+.3f} (n={data['n_samples']})")

    # 4g: Mean absolute error comparison
    xgb_mae = float(np.mean(np.abs(xgb_preds_all - y)))
    bert_mae = float(np.mean(np.abs(bert_preds_all - y)))
    log(f"\n  Mean Absolute Error: XGBoost={xgb_mae:.3f}, BERT={bert_mae:.3f}")

    diagnostic_results = {
        "probing_r2": f"{mean_r2:.4f} +/- {std_r2:.4f}",
        "probing_r2_mean": mean_r2,
        "group_probing_r2": group_r2,
        "agreement": {
            "both_correct_pct": float(100*both_correct.mean()),
            "xgb_only_pct": float(100*xgb_only.mean()),
            "bert_only_pct": float(100*bert_only.mean()),
            "both_wrong_pct": float(100*both_wrong.mean()),
        },
        "bert_advantage_by_level": bert_advantage_by_level,
        "top_probing_features": coef_df.head(10)[["feature", "ridge_coef_abs"]].to_dict("records"),
        "xgb_per_class_f1": {CEFR_LEVELS[i]: float(xgb_per_class[i]) for i in range(6)},
        "bert_per_class_f1": {CEFR_LEVELS[i]: float(bert_per_class[i]) for i in range(6)},
        "xgb_mae": xgb_mae,
        "bert_mae": bert_mae,
    }

    return diagnostic_results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_KEY)
    args = parser.parse_args()
    dataset_key = args.dataset

    start_time = time.time()

    # Per-dataset output dirs
    global RESULTS_DIR, FIGURES_DIR, MODELS_DIR

    RESULTS_DIR = results_dir(dataset_key)
    FIGURES_DIR = figures_dir(dataset_key)
    MODELS_DIR = os.path.join(RESULTS_DIR, "models")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    log("Loading data...")
    df = load_data(dataset_key=dataset_key)
    splits = get_cv_splits(df)

    # Load cached features
    log("\nLoading features...")
    features_path = os.path.join(RESULTS_DIR, "features.csv")
    groups_path = os.path.join(RESULTS_DIR, "feature_groups.json")

    if os.path.exists(features_path) and os.path.exists(groups_path):
        log("Loading cached features...")
        features_df = pd.read_csv(features_path)
        with open(groups_path) as f:
            feature_groups = json.load(f)
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        log("ERROR: Features not cached. Run feature_extraction.py first.")
        return

    log(f"Features: {features_df.shape[1]} total")
    if len(features_df) != len(df):
        raise ValueError(
            f"Feature/Data mismatch for dataset={dataset_key}: "
            f"features_df has {len(features_df)} rows, df has {len(df)} rows. "
            f"Did you run feature_extraction.py --dataset {dataset_key}?"
        )

    # Experiment 1: EDA
    corr_df, group_corr_df = run_eda(df, features_df, feature_groups, dataset_key)

    # Experiment 2: Interpretable Models
    feature_results, ablation_results, imp_df = run_interpretable_models(
        features_df, df["label"], splits, feature_groups
    )
    tfidf_svm_results = run_tfidf_linear_svm(df, splits)

    # Experiment 3: Neural models
    bert_results, bert_logits = run_bert(df, splits)
    dan_results = run_dan(df, splits)

    # Experiment 4: Diagnostics
    diagnostic_results = run_diagnostics(
        features_df, df["label"], bert_logits, splits, feature_groups,
        feature_results, bert_results
    )

    elapsed = time.time() - start_time

    # Save results
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()
                    if k not in ("fold_results", "y_pred", "y_true", "y_test", "logits", "confusion_matrix")}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary = {
        "experiment_info": {
            "dataset": dataset_key,
            "n_samples": len(df),
            "n_features": features_df.shape[1],
            "n_folds": N_FOLDS,
            "seed": SEED,
            "elapsed_seconds": elapsed,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "feature_models": clean_for_json(feature_results),
        "tfidf_linear_svm": clean_for_json(tfidf_svm_results),
        "bert": clean_for_json(bert_results),
        "dan": clean_for_json(dan_results),
        "ablation": clean_for_json(ablation_results),
        "diagnostics": clean_for_json(diagnostic_results),
    }

    with open(os.path.join(RESULTS_DIR, "experiment_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Summary
    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)
    log(f"\n{'Model':<25} {'Macro F1':<22} {'Accuracy':<22} {'Adj. Acc':<12}")
    log("-"*80)
    for model_name in ["Majority", "Full_LogReg", "Full_RandomForest", "Full_XGBoost"]:
        if model_name in feature_results:
            r = feature_results[model_name]
            f1_str = r.get("macro_f1", f"{r.get('macro_f1_mean', 0):.4f}")
            acc_str = r.get("accuracy", "N/A")
            adj_str = r.get("adjacent_acc", "N/A")
            log(f"{model_name:<25} {f1_str:<22} {acc_str:<22} {adj_str:<12}")
    r = tfidf_svm_results
    log(f"{'TFIDF_LinearSVM':<25} {r['macro_f1']:<22} {r['accuracy']:<22} {r['adjacent_acc']:<12}")
    r = bert_results
    log(f"{'BERT':<25} {r['macro_f1']:<22} {r['accuracy']:<22} {r['adjacent_acc']:<12}")
    r = dan_results
    log(f"{'DAN':<25} {r['macro_f1']:<22} {r['accuracy']:<22} {r['adjacent_acc']:<12}")

    log(f"\nProbing R2 (features -> BERT): {diagnostic_results['probing_r2']}")
    log(f"Total time: {elapsed/60:.1f} minutes")

    # Model comparison plot
    model_names = []
    f1_means = []
    f1_stds = []
    for name in ["Full_LogReg", "Full_RandomForest", "Full_XGBoost"]:
        if name in feature_results:
            model_names.append(name.replace("Full_", ""))
            f1_means.append(feature_results[name]["macro_f1_mean"])
            f1_stds.append(feature_results[name]["macro_f1_std"])
    model_names.append("TFIDF+SVM")
    f1_means.append(tfidf_svm_results["macro_f1_mean"])
    f1_stds.append(tfidf_svm_results["macro_f1_std"])
    model_names.append("BERT")
    f1_means.append(bert_results["macro_f1_mean"])
    f1_stds.append(bert_results["macro_f1_std"])
    model_names.append("DAN")
    f1_means.append(dan_results["macro_f1_mean"])
    f1_stds.append(dan_results["macro_f1_std"])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#2ecc71", "#f39c12", "#34495e", "#e74c3c", "#9b59b6"]
    bars = ax.bar(model_names, f1_means, yerr=f1_stds, capsize=5,
                  color=colors[:len(model_names)], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Model Comparison: Macro F1 on CEFR-SP (5-fold CV)")
    ax.set_ylim(0, max(f1_means) * 1.3)
    for bar, mean in zip(bars, f1_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{mean:.3f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # Ablation plot
    ablation_names = []
    ablation_f1s = []
    for key, val in ablation_results.items():
        if key.startswith("Only_"):
            ablation_names.append(key.replace("Only_", ""))
            ablation_f1s.append(val["macro_f1_mean"])

    if ablation_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_idx = np.argsort(ablation_f1s)[::-1]
        sorted_names = [ablation_names[i] for i in sorted_idx]
        sorted_f1s = [ablation_f1s[i] for i in sorted_idx]
        bars = ax.bar(sorted_names, sorted_f1s,
                      color=sns.color_palette("viridis", len(sorted_names)),
                      edgecolor="black", linewidth=0.5)
        ax.axhline(y=feature_results["Full_XGBoost"]["macro_f1_mean"],
                    color="red", linestyle="--", label="Full XGBoost model")
        ax.axhline(y=feature_results["Majority"]["macro_f1_mean"],
                    color="gray", linestyle=":", label="Majority baseline")
        ax.set_ylabel("Macro F1")
        ax.set_title("Feature Group Ablation: Individual Groups vs. Full Model")
        ax.legend()
        for bar, f1 in zip(bars, sorted_f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{f1:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "ablation_individual_groups.png"), dpi=150)
        plt.close()

    log("\nAll experiments complete! Results saved to results/ and figures/")
    return summary


if __name__ == "__main__":
    summary = main()
