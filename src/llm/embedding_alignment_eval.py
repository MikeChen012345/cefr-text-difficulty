from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CEFR_TO_INT = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two embedding models via CCA alignment and retrieval metrics."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("datasets/cefr_sp_en/train.csv"),
        help="Path to CEFR dataset CSV.",
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model A.",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="sentence-transformers/paraphrase-albert-small-v2",
        help="Embedding model B.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max number of rows to evaluate (0 means all).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio for retrieval/CCA reporting.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieval cutoff.",
    )
    parser.add_argument(
        "--cca-components",
        type=int,
        default=20,
        help="Max number of CCA components.",
    )
    parser.add_argument(
        "--cca-pca-dim",
        type=int,
        default=128,
        help="PCA dim per space before CCA (0 disables PCA).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding encoding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/cefr_sp_en"),
        help="Directory to save metrics.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/cefr_sp_en"),
        help="Directory to save figures.",
    )
    return parser.parse_args()


def load_dataset(path: Path, max_samples: int, seed: int) -> Tuple[List[str], np.ndarray]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            level = str(row.get("cefr_level", "")).strip()
            text = str(row.get("text", "")).strip()
            if level not in CEFR_TO_INT or not text:
                continue
            texts.append(text)
            labels.append(CEFR_TO_INT[level])

    if max_samples > 0 and len(texts) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(texts), size=max_samples, replace=False)
        idx = np.sort(idx)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]

    return texts, np.asarray(labels, dtype=np.int64)


def embed_texts(model_name: str, texts: List[str], batch_size: int) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32, copy=False)


def retrieval_metrics(
    db_emb: np.ndarray,
    db_labels: np.ndarray,
    query_emb: np.ndarray,
    query_labels: np.ndarray,
    top_k: int,
) -> Dict[str, float]:
    top_k = int(min(top_k, db_emb.shape[0]))
    sims = np.matmul(query_emb, db_emb.T)
    top_idx = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]

    top1_hits = 0
    topk_hits = 0
    mrr = 0.0

    for i in range(query_emb.shape[0]):
        q_label = query_labels[i]
        cand = top_idx[i]
        cand = cand[np.argsort(-sims[i, cand])]
        cand_labels = db_labels[cand]

        if cand_labels[0] == q_label:
            top1_hits += 1
        if np.any(cand_labels == q_label):
            topk_hits += 1
            first_pos = int(np.where(cand_labels == q_label)[0][0]) + 1
            mrr += 1.0 / first_pos

    n = query_emb.shape[0]
    return {
        "top1_label_acc": float(top1_hits / n),
        f"top{top_k}_label_hit": float(topk_hits / n),
        f"mrr@{top_k}": float(mrr / n),
    }


def fit_cca_and_correlations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels_test: np.ndarray,
    cca_components: int,
    cca_pca_dim: int,
    seed: int,
) -> Dict[str, object]:
    sx = StandardScaler()
    sy = StandardScaler()
    x_train_s = sx.fit_transform(x_train)
    y_train_s = sy.fit_transform(y_train)
    x_test_s = sx.transform(x_test)
    y_test_s = sy.transform(y_test)

    if cca_pca_dim > 0:
        px_dim = min(cca_pca_dim, x_train_s.shape[1], x_train_s.shape[0] - 1)
        py_dim = min(cca_pca_dim, y_train_s.shape[1], y_train_s.shape[0] - 1)
        px = PCA(n_components=px_dim, random_state=seed)
        py = PCA(n_components=py_dim, random_state=seed)
        x_train_s = px.fit_transform(x_train_s)
        y_train_s = py.fit_transform(y_train_s)
        x_test_s = px.transform(x_test_s)
        y_test_s = py.transform(y_test_s)

    n_comp = min(
        cca_components,
        x_train_s.shape[1],
        y_train_s.shape[1],
        x_train_s.shape[0] - 1,
    )
    if n_comp < 1:
        raise ValueError("Not enough samples/features for CCA.")

    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(x_train_s, y_train_s)
    u_test, v_test = cca.transform(x_test_s, y_test_s)

    corrs = []
    for i in range(n_comp):
        c = np.corrcoef(u_test[:, i], v_test[:, i])[0, 1]
        if not np.isfinite(c):
            c = 0.0
        corrs.append(float(c))

    label_corr_u = []
    label_corr_v = []
    y_ord = labels_test.astype(np.float32)
    for i in range(n_comp):
        cu = np.corrcoef(u_test[:, i], y_ord)[0, 1]
        cv = np.corrcoef(v_test[:, i], y_ord)[0, 1]
        label_corr_u.append(float(0.0 if not np.isfinite(cu) else cu))
        label_corr_v.append(float(0.0 if not np.isfinite(cv) else cv))

    top5 = float(np.mean(corrs[: min(5, len(corrs))]))
    top10 = float(np.mean(corrs[: min(10, len(corrs))]))
    return {
        "n_components": int(n_comp),
        "canonical_correlations": corrs,
        "mean_top5": top5,
        "mean_top10": top10,
        "label_corr_u": label_corr_u,
        "label_corr_v": label_corr_v,
    }


def save_retrieval_plot(
    model_names: List[str],
    retrieval_rows: List[Dict[str, float]],
    out_path: Path,
    top_k: int,
) -> None:
    metrics = ["top1_label_acc", f"top{top_k}_label_hit", f"mrr@{top_k}"]
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, metric in enumerate(metrics):
        vals = [row[metric] for row in retrieval_rows]
        ax.bar(x + (i - 1) * width, vals, width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=10, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Embedding Retrieval Performance (CEFR-label consistency)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_cca_plot(corrs: List[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(1, len(corrs) + 1)
    ax.plot(xs, corrs, marker="o", linewidth=2)
    ax.set_xlabel("Canonical Component")
    ax.set_ylabel("Correlation")
    ax.set_title("CCA Alignment Between Embedding Spaces")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    texts, labels = load_dataset(args.dataset_path, args.max_samples, args.seed)
    train_idx, test_idx = train_test_split(
        np.arange(len(texts)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    print(f"Loaded {len(texts)} samples ({len(train_texts)} train / {len(test_texts)} test)")
    print(f"Encoding model A: {args.model_a}")
    a_train = embed_texts(args.model_a, train_texts, args.batch_size)
    a_test = embed_texts(args.model_a, test_texts, args.batch_size)

    print(f"Encoding model B: {args.model_b}")
    b_train = embed_texts(args.model_b, train_texts, args.batch_size)
    b_test = embed_texts(args.model_b, test_texts, args.batch_size)

    ret_a = retrieval_metrics(a_train, y_train, a_test, y_test, args.top_k)
    ret_b = retrieval_metrics(b_train, y_train, b_test, y_test, args.top_k)

    cca_out = fit_cca_and_correlations(
        x_train=a_train,
        y_train=b_train,
        x_test=a_test,
        y_test=b_test,
        labels_test=y_test,
        cca_components=args.cca_components,
        cca_pca_dim=args.cca_pca_dim,
        seed=args.seed,
    )

    retrieval_csv = args.results_dir / "embedding_retrieval_metrics.csv"
    with retrieval_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "top1_label_acc", f"top{args.top_k}_label_hit", f"mrr@{args.top_k}"],
        )
        writer.writeheader()
        writer.writerow({"model": args.model_a, **ret_a})
        writer.writerow({"model": args.model_b, **ret_b})

    summary_json = args.results_dir / "embedding_alignment_summary.json"
    summary = {
        "dataset_path": str(args.dataset_path),
        "n_samples": len(texts),
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "model_a": args.model_a,
        "model_b": args.model_b,
        "retrieval_metrics": {
            "model_a": ret_a,
            "model_b": ret_b,
            "top_k": args.top_k,
        },
        "cca": cca_out,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    cca_fig = args.figures_dir / "embedding_cca_correlations.png"
    ret_fig = args.figures_dir / "embedding_retrieval_comparison.png"
    save_cca_plot(cca_out["canonical_correlations"], cca_fig)
    save_retrieval_plot(
        [short_name(args.model_a), short_name(args.model_b)],
        [ret_a, ret_b],
        ret_fig,
        top_k=args.top_k,
    )

    print("\n=== Retrieval (label-consistency) ===")
    print(f"{short_name(args.model_a)}: {ret_a}")
    print(f"{short_name(args.model_b)}: {ret_b}")
    print("\n=== CCA Alignment ===")
    print(f"n_components={cca_out['n_components']}")
    print(f"mean_top5={cca_out['mean_top5']:.4f}, mean_top10={cca_out['mean_top10']:.4f}")
    print(f"Saved: {retrieval_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {cca_fig}")
    print(f"Saved: {ret_fig}")


if __name__ == "__main__":
    main()
