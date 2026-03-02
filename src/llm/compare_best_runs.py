from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

if Path.cwd().name == "llm":
    sys.path.append("..")
elif Path.cwd().name == "src":
    sys.path.append(".")
elif Path.cwd().name == "cefr-text-difficulty":
    sys.path.append("./src")

from config import results_dir, figures_dir


def _to_int(value: object, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            return int(value.strip())
        return default
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        return default
    except (TypeError, ValueError):
        return default


def _latest_best_summary(results_path: Path, method: str) -> Dict[str, object] | None:
    pattern = f"{method}_best_model*_summary.json"
    candidates = list(results_path.glob(pattern))
    if not candidates:
        return None
    print(candidates)
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    with latest.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["summary_file"] = str(latest)
    return data


def _collect_best_summaries(dataset_key: str) -> pd.DataFrame:
    results_path = Path(results_dir(dataset_key))
    methods = ["llm", "rag", "multi_agent", "tool"]

    rows: List[Dict[str, object]] = []
    for method in methods:
        summary = _latest_best_summary(results_path, method)
        if summary is None:
            continue
        rows.append(
            {
                "method": method,
                "model_idx": _to_int(summary.get("model_idx", -1), default=-1),
                "model_name": str(summary.get("model_name", "")),
                "accuracy": _to_float(summary.get("accuracy", 0.0)),
                "macro_f1": _to_float(summary.get("macro_f1", 0.0)),
                "adjacent_accuracy": _to_float(summary.get("adjacent_accuracy", 0.0)),
                "average_token_usage": _to_float(summary.get("average_token_usage", 0.0)),
                "average_elapsed_time": _to_float(summary.get("average_elapsed_time", 0.0)),
                "n_samples": _to_int(summary.get("n_samples", 0)),
                "summary_file": str(summary.get("summary_file", "")),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    method_order = ["llm", "rag", "multi_agent", "tool"]
    df["method_order"] = df["method"].map({k: i for i, k in enumerate(method_order)})
    df = df.sort_values("method_order").drop(columns=["method_order"]).reset_index(drop=True)
    return df


def _plot_best_run_comparison(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("adjacent_accuracy", "Adjacent accuracy"),
        ("average_token_usage", "Avg. token usage"),
        ("average_elapsed_time", "Avg. elapsed time (s)"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.5))
    method_labels = df["method"].tolist()
    x_positions = list(range(len(method_labels)))
    color_map = {
        method: plt.get_cmap("tab10")(idx)
        for idx, method in enumerate(method_labels)
    }
    bar_colors = [color_map[m] for m in method_labels]

    for ax, (metric_col, metric_title) in zip(axes, metrics, strict=False):
        y_values = df[metric_col].astype(float).tolist()
        ax.bar(x_positions, y_values, color=bar_colors)
        ax.set_title(metric_title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(method_labels, rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    handles = [
        Line2D([0], [0], color=color_map[m], lw=6)
        for m in method_labels
    ]
    fig.legend(handles, method_labels, loc="upper center", ncol=len(method_labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main(dataset_key: str = "readme_en") -> None:
    df = _collect_best_summaries(dataset_key=dataset_key)
    if df.empty:
        print(f"No best-run summary JSON files found in: {results_dir(dataset_key)}")
        return

    if "llm" not in set(df["method"].tolist()):
        print("LLM baseline summary JSON is missing. Run llm.run_best_model(...) first.")
        return

    output_results_dir = Path(results_dir(dataset_key))
    output_figures_dir = Path(figures_dir(dataset_key))
    comparison_csv = output_results_dir / "best_run_comparison_vs_llm.csv"
    comparison_json = output_results_dir / "best_run_comparison_vs_llm.json"
    comparison_fig = output_figures_dir / "best_run_comparison_vs_llm.png"

    df.to_csv(comparison_csv, index=False)
    with comparison_json.open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    _plot_best_run_comparison(df, comparison_fig)

    print(f"Saved comparison table CSV to: {comparison_csv}")
    print(f"Saved comparison table JSON to: {comparison_json}")
    print(f"Saved comparison figure to: {comparison_fig}")


if __name__ == "__main__":
    dataset_keys = ["readme_en", "cefr_sp_en"]
    for dataset_key in dataset_keys:
        main(dataset_key=dataset_key)
