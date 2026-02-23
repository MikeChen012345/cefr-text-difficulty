"""Extract linguistic features from CEFR-SP sentences.

Feature groups:
1. Lexical: word frequency, TTR, word length, difficult word ratio
2. Syntactic: dependency depth/distance, POS distributions, sentence length
3. Surprisal: GPT-2 per-word surprisal statistics
4. Readability: Flesch-Kincaid, ARI, Coleman-Liau, SMOG (traditional baselines)
"""
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import spacy
import textstat
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

from config import SEED, DEVICE, results_dir, DEFAULT_DATASET_KEY
from data_loader import load_data

np.random.seed(SEED)


# ── Readability features ───────────────────────────────────────────────
def extract_readability_features(texts):
    """Traditional readability formulas via textstat."""
    records = []
    for text in tqdm(texts, desc="Readability"):
        rec = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "smog_index": textstat.smog_index(text),
            "gunning_fog": textstat.gunning_fog(text),
            "dale_chall": textstat.dale_chall_readability_score(text),
        }
        records.append(rec)
    return pd.DataFrame(records)


# ── Lexical features ──────────────────────────────────────────────────
def extract_lexical_features(docs):
    """Lexical features from spaCy docs."""
    records = []
    for doc in tqdm(docs, desc="Lexical"):
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        n_tokens = len(tokens) if tokens else 1

        # Word lengths
        word_lengths = [len(t.text) for t in tokens] if tokens else [0]

        # Type-token ratio
        types = set(t.lower_ for t in tokens)
        ttr = len(types) / n_tokens if n_tokens > 0 else 0

        # Corrected TTR (Guiraud's index)
        guiraud = len(types) / np.sqrt(n_tokens) if n_tokens > 0 else 0

        # Syllable count per word (approximate)
        syll_counts = [textstat.syllable_count(t.text) for t in tokens] if tokens else [0]

        rec = {
            "n_tokens": n_tokens,
            "n_types": len(types),
            "ttr": ttr,
            "guiraud_index": guiraud,
            "mean_word_length": np.mean(word_lengths),
            "max_word_length": np.max(word_lengths) if word_lengths else 0,
            "std_word_length": np.std(word_lengths),
            "mean_syllables": np.mean(syll_counts),
            "max_syllables": np.max(syll_counts) if syll_counts else 0,
            "prop_long_words": sum(1 for w in word_lengths if w > 6) / n_tokens,
            "prop_rare_words": sum(1 for t in tokens if t.is_oov) / n_tokens,
        }
        records.append(rec)
    return pd.DataFrame(records)


# ── Syntactic features ────────────────────────────────────────────────
def extract_syntactic_features(docs):
    """Syntactic complexity from dependency parses."""
    records = []
    for doc in tqdm(docs, desc="Syntactic"):
        sents = list(doc.sents)
        n_sents = len(sents) if sents else 1

        # Sentence length stats
        sent_lengths = [len(s) for s in sents]

        # Dependency tree depth
        def tree_depth(token):
            depth = 0
            while token.head != token:
                depth += 1
                token = token.head
            return depth

        depths = [tree_depth(t) for t in doc]

        # Dependency distances (linear distance between head and dependent)
        dep_distances = [abs(t.i - t.head.i) for t in doc if t.head != t]

        # POS tag distribution
        pos_counts = {}
        for t in doc:
            pos_counts[t.pos_] = pos_counts.get(t.pos_, 0) + 1
        n_tokens = len(doc) if len(doc) > 0 else 1

        # Subordinate clauses (approx: count "mark" and "advcl" relations)
        n_subclauses = sum(1 for t in doc if t.dep_ in ("advcl", "ccomp", "xcomp", "acl", "relcl"))

        rec = {
            "n_sentences": n_sents,
            "mean_sent_length": np.mean(sent_lengths),
            "max_sent_length": np.max(sent_lengths) if sent_lengths else 0,
            "mean_tree_depth": np.mean(depths) if depths else 0,
            "max_tree_depth": np.max(depths) if depths else 0,
            "mean_dep_distance": np.mean(dep_distances) if dep_distances else 0,
            "max_dep_distance": np.max(dep_distances) if dep_distances else 0,
            "n_subclauses": n_subclauses,
            "subclause_ratio": n_subclauses / n_sents,
            # POS proportions
            "prop_noun": pos_counts.get("NOUN", 0) / n_tokens,
            "prop_verb": pos_counts.get("VERB", 0) / n_tokens,
            "prop_adj": pos_counts.get("ADJ", 0) / n_tokens,
            "prop_adv": pos_counts.get("ADV", 0) / n_tokens,
            "prop_adp": pos_counts.get("ADP", 0) / n_tokens,
            "prop_conj": (pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0)) / n_tokens,
            "n_unique_pos": len(pos_counts),
        }
        records.append(rec)
    return pd.DataFrame(records)


# ── GPT-2 Surprisal features ─────────────────────────────────────────
def extract_surprisal_features(texts, batch_size=32):
    """Compute per-sentence surprisal statistics using GPT-2."""
    print("Loading GPT-2 for surprisal computation...")
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    records = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Surprisal"):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits  # (B, T, V)

        for j in range(len(batch_texts)):
            input_ids = encodings["input_ids"][j]
            attn_mask = encodings["attention_mask"][j]
            seq_logits = logits[j]

            # Surprisal = -log2(P(token | context))
            log_probs = torch.log_softmax(seq_logits, dim=-1)

            # Shift: predict token t from context [0..t-1]
            # surprisal for token at position t is -log_probs[t-1, input_ids[t]]
            token_surprisals = []
            for t in range(1, attn_mask.sum().item()):
                token_id = input_ids[t].item()
                if token_id == tokenizer.eos_token_id:
                    continue
                s = -log_probs[t-1, token_id].item() / np.log(2)  # convert to bits
                token_surprisals.append(s)

            if not token_surprisals:
                token_surprisals = [0.0]

            arr = np.array(token_surprisals)
            records.append({
                "mean_surprisal": arr.mean(),
                "max_surprisal": arr.max(),
                "min_surprisal": arr.min(),
                "std_surprisal": arr.std(),
                "median_surprisal": np.median(arr),
                "surprisal_range": arr.max() - arr.min(),
                "perplexity": 2 ** arr.mean(),
            })

    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(records)


# ── LFTK features (if available) ─────────────────────────────────────
def extract_lftk_features(docs):
    """Extract features using LFTK library."""
    try:
        import lftk
        extractor = lftk.Extractor(docs=docs)
        extractor.customize(stop_words=True, punctuations=False, round_decimal=3)
        features = extractor.extract()
        df = pd.DataFrame(features)
        # Drop columns that are all NaN
        df = df.dropna(axis=1, how="all")
        print(f"LFTK extracted {df.shape[1]} features")
        return df
    except Exception as e:
        print(f"LFTK extraction failed: {e}")
        return pd.DataFrame()


# ── Main extraction pipeline ─────────────────────────────────────────
def extract_all_features(df, use_lftk=True, use_surprisal=True):
    """Extract all feature groups and return combined DataFrame."""
    texts = df["text"].tolist()

    # Process with spaCy
    print("Processing with spaCy...")
    nlp = spacy.load("en_core_web_sm")
    docs = list(nlp.pipe(texts, batch_size=256))

    # Extract each feature group
    feat_readability = extract_readability_features(texts)
    feat_lexical = extract_lexical_features(docs)
    feat_syntactic = extract_syntactic_features(docs)

    if use_surprisal:
        feat_surprisal = extract_surprisal_features(texts)
    else:
        feat_surprisal = pd.DataFrame()

    if use_lftk:
        feat_lftk = extract_lftk_features(docs)
    else:
        feat_lftk = pd.DataFrame()

    # Define feature group membership
    feature_groups = {
        "readability": list(feat_readability.columns),
        "lexical": list(feat_lexical.columns),
        "syntactic": list(feat_syntactic.columns),
    }
    if not feat_surprisal.empty:
        feature_groups["surprisal"] = list(feat_surprisal.columns)
    if not feat_lftk.empty:
        # Exclude features already in other groups
        existing = set()
        for cols in feature_groups.values():
            existing.update(cols)
        lftk_cols = [c for c in feat_lftk.columns if c not in existing]
        feature_groups["lftk_extra"] = lftk_cols

    # Combine all features
    all_dfs = [feat_readability, feat_lexical, feat_syntactic]
    if not feat_surprisal.empty:
        all_dfs.append(feat_surprisal)
    if not feat_lftk.empty:
        # Only add non-duplicate columns
        existing_cols = set()
        for d in all_dfs:
            existing_cols.update(d.columns)
        lftk_unique = feat_lftk[[c for c in feat_lftk.columns if c not in existing_cols]]
        if not lftk_unique.empty:
            all_dfs.append(lftk_unique)

    features_df = pd.concat(all_dfs, axis=1)

    # Handle infinities and NaN
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)

    print(f"\nTotal features: {features_df.shape[1]}")
    print(f"Feature groups: {json.dumps({k: len(v) for k, v in feature_groups.items()}, indent=2)}")

    return features_df, feature_groups


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_KEY)
    parser.add_argument("--no_surprisal", action="store_true")
    parser.add_argument("--no_lftk", action="store_true")
    args = parser.parse_args()

    df = load_data(dataset_key=args.dataset)

    # Ignore surprisal features for now
    # features_df, feature_groups = extract_all_features(
    #     df,
    #     use_lftk=not args.no_lftk,
    #     use_surprisal=not args.no_surprisal
    # )
    features_df, feature_groups = extract_all_features(df, use_lftk=False, use_surprisal=False)

    out_dir = results_dir(args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    features_df.to_csv(os.path.join(out_dir, "features.csv"), index=False)
    with open(os.path.join(out_dir, "feature_groups.json"), "w") as f:
        json.dump(feature_groups, f, indent=2)

    print(f"\nSaved {features_df.shape[1]} features for {features_df.shape[0]} samples to {out_dir}")
