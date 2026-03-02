from __future__ import annotations

import yaml
import sys
import os

if os.getcwd().endswith("llm"):
    sys.path.append("..")  # Add parent directory to path for imports
    config = yaml.safe_load(open("config.yaml", "r"))
elif os.getcwd().endswith("src"):
    sys.path.append(".")  # Ensure current directory is in path for imports
    config = yaml.safe_load(open("llm/config.yaml", "r"))
elif os.getcwd().endswith("cefr-text-difficulty"):
    # else if running from project root
    sys.path.append("./src")  # Add src directory to path for imports
    config = yaml.safe_load(open("src/llm/config.yaml", "r"))
else:
    raise ValueError("Unexpected working directory; cannot locate config.yaml")

import logging
from typing import Dict, List, Tuple, Optional
import time
import json
import importlib.util

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from data_loader import load_data_split
from prompt import build_system_prompt_multi_agent, build_system_prompt_critique
from config import results_dir, figures_dir

logger = logging.getLogger(__name__)

# Reuse helper implementations from llm.py (single source of truth)
llm_file = Path(__file__).with_name("llm.py")
llm_spec = importlib.util.spec_from_file_location("llm_helpers_module", llm_file)
if llm_spec is None or llm_spec.loader is None:
    raise ModuleNotFoundError(f"Unable to load helper functions from {llm_file}")
llm_module = importlib.util.module_from_spec(llm_spec)
llm_spec.loader.exec_module(llm_module)
_record_usage = llm_module._record_usage
_message_text = llm_module._message_text
_extract_final_answer = llm_module._extract_final_answer
_compute_metrics = llm_module._compute_metrics

if not config.get("api_key") or not config.get("api_key").strip():  # None or empty string
    logger.warning("No API key found in config.yaml; will attempt to retrieve from inference service")
    from utils.inference_auth_token import get_access_token  # for inference service
    config["api_key"] = get_access_token()
    config["model_provider"] = "openai"  # openai-compatible API

# Set the dataset to use
DATASET_KEY = "cefr_sp_en" # "readme_en" # 

# Setup directories for results and figures based on dataset
RESULTS_DIR = results_dir(DATASET_KEY)
FIGURES_DIR = figures_dir(DATASET_KEY)


def _extract_model_size(model_name: str) -> Optional[float]:
    size_pattern = re.compile(r"(\d+(?:\.\d+)?)([MB])", re.IGNORECASE)
    match = size_pattern.search(model_name)
    if match:
        number_str, unit = match.groups()
        try:
            number = float(number_str)
            if unit.upper() == "M":
                return number / 1000
            if unit.upper() == "B":
                return number
        except ValueError:
            pass
    return None


def _build_model(model_idx: int) -> BaseChatModel:
    print(f"Building model with {config['models'][model_idx]}, temperature={config['temperature']}, timeout={config['timeout']}s")
    print("=" * 50)
    return init_chat_model(
        model=config["models"][model_idx],
        model_provider=config["model_provider"],
        temperature=config["temperature"],
        timeout=config["timeout"],
        max_retries=config["max_retries"],
        max_tokens=config["max_tokens"],
        base_url=config["base_url"],
        api_key=config["api_key"],
    )


def multi_agent_llm(
    primary_model: BaseChatModel,
    critique_model: BaseChatModel,
    text: str,
    cot: bool = False,
    few_shots: int = 0,
) -> tuple[str, str, int, float, str, str, str]:
    """Two-agent placeholder flow (no RAG):
    1) primary agent predicts CEFR
    2) critique agent comments on primary output
    """
    now = time.time()
    text = text.strip()
    if not text:
        raise ValueError("Input text is empty.")

    #### Primary agent step: predict CEFR level and rationale
    primary_system_prompt = build_system_prompt_multi_agent(
        dataset_key=DATASET_KEY,
        n=few_shots,
        cot=cot,
    )

    token_usage = 0

    primary_conversation: List[HumanMessage | AIMessage | SystemMessage] = [
        SystemMessage(content=primary_system_prompt),
        HumanMessage(content=f"Text:\n{text}"),
    ]
    initial_message = primary_model.invoke(primary_conversation)
    token_usage += _record_usage(initial_message)
    initial_text = _message_text(initial_message)
    initial_answer, initial_rationale = _extract_final_answer(initial_text)

    #### Critique agent step: provide feedback on primary agent's rationale and prediction
    critique_system_prompt = build_system_prompt_critique()

    critique_conversation: List[HumanMessage | AIMessage | SystemMessage] = [
        SystemMessage(content=critique_system_prompt),
        HumanMessage(
            content=(
                f"Text:\n{text}\n\n"
                f"First model rationale:\n{initial_rationale}\n\n"
                f"First model prediction:\n{initial_answer}"
            )
        ),
    ]
    critique_message = critique_model.invoke(critique_conversation)
    token_usage += _record_usage(critique_message)
    critique_text = _message_text(critique_message)

    #### Primary agent step again: revise CEFR level prediction based on critique feedback
    refined_conversation: List[HumanMessage | AIMessage | SystemMessage] = primary_conversation + [
        HumanMessage(
            content= (
                f"Critique of your rationale and prediction:\n{critique_text}\n\n"
                f"Please provide a final CEFR level assessment for the input text, taking into account the critique's feedback. Be sure to provide a rationale for your final assessment as well."
                f"Follow the same response format as before: <rationale>#### <CEFR level>"
            )
        ),
    ]

    final_message = primary_model.invoke(refined_conversation)
    token_usage += _record_usage(final_message)
    final_text = _message_text(final_message)
    final_answer, final_rationale = _extract_final_answer(final_text)

    return final_answer, final_rationale, token_usage, time.time() - now, initial_answer, initial_rationale, critique_text


def evaluate_model_on_test_set(
    test_df: pd.DataFrame,
    primary_model: BaseChatModel,
    critique_model: BaseChatModel,
    model_idx: int,
    cot: bool,
    few_shots: int,
) -> List[Dict[str, str | int | float]]:
    parallel_cfg = config.get("parallelism", {}) if isinstance(config, dict) else {}
    max_workers = int(parallel_cfg.get("max_workers", 1) or 1)

    def _eval_one(i: int) -> Dict[str, str | int | float]:
        row = test_df.iloc[i]
        text = str(row["text"])
        true_label = row["cefr_level"]
        predicted_label, rationale, token_usage, elapsed_time, initial_label, initial_rationale, critique = multi_agent_llm(
            primary_model,
            critique_model,
            text,
            cot=cot,
            few_shots=few_shots,
        )
        return {
            "text": text,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "initial_label": initial_label,
            "initial_rationale": initial_rationale,
            "rationale": rationale,
            "critique": critique,
            "token_usage": token_usage,
            "elapsed_time": elapsed_time,
        }

    results_by_idx: Dict[int, Dict[str, str | int | float]] = {}
    if max_workers <= 1:
        for i in tqdm(range(len(test_df)), desc="Evaluating"):
            results_by_idx[i] = _eval_one(i)
    else:
        logger.info("Evaluating %d samples with max_workers=%d", len(test_df), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_eval_one, i): i for i in range(len(test_df))}
            for fut in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="Evaluating",
            ):
                i = future_to_idx[fut]
                results_by_idx[i] = fut.result()

    results = [results_by_idx[i] for i in range(len(test_df))]

    run_results_dir = Path(RESULTS_DIR)
    run_results_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_results_dir / f"multi_agent_model{model_idx}_CoT{cot}_Shot{few_shots}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


_RUN_FILE_RE = re.compile(
    r"^multi_agent_model(?P<model_idx>\d+)_CoT(?P<cot>True|False)_Shot(?P<shots>\d+)\.json$"
)


def summarize_results(results_dir: Path | None = None) -> pd.DataFrame:
    results_dir = results_dir or Path(RESULTS_DIR)
    rows: List[Dict[str, str | int | float | bool]] = []

    for file in sorted(results_dir.glob("multi_agent_model*_CoT*_Shot*.json")):
        match = _RUN_FILE_RE.match(file.name)
        if not match:
            continue

        model_idx = int(match.group("model_idx"))
        cot = match.group("cot") == "True"
        few_shots = int(match.group("shots"))

        with file.open("r", encoding="utf-8") as f:
            results = json.load(f)

        accuracy, macro_f1, adjacent_accuracy, avg_token_usage, avg_elapsed_time = _compute_metrics(results)
        model_name = ""
        try:
            model_name = config.get("models", [])[model_idx]
        except Exception:
            model_name = ""

        rows.append(
            {
                "model_idx": model_idx,
                "model_name": model_name,
                "cot": cot,
                "few_shots": few_shots,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "adjacent_accuracy": adjacent_accuracy,
                "average_token_usage": avg_token_usage,
                "average_elapsed_time": avg_elapsed_time,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "model_idx",
            "model_name",
            "cot",
            "few_shots",
            "accuracy",
            "macro_f1",
            "adjacent_accuracy",
            "average_token_usage",
            "average_elapsed_time",
        ],
    )


def save_summary_to_file(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    results_dir = results_dir or Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (results_dir / "multi_agent_results_summary.csv")
    df.to_csv(out_path, index=False)
    return out_path


def plot_model_size_summary(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    few_shots: Optional[int] = None,
):
    results_dir = results_dir or Path(RESULTS_DIR)
    out_dir = out_dir or Path(FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df: pd.DataFrame
    if df is None:
        summary_df = summarize_results(results_dir=results_dir)
    else:
        summary_df = df

    if summary_df.empty:
        logger.warning("No summary rows found; skipping plots.")
        return []

    if few_shots is None:
        unique_shots = sorted({int(x) for x in summary_df["few_shots"].unique().tolist()})
        if len(unique_shots) > 1:
            few_shots = 0 if 0 in unique_shots else unique_shots[0]

    if few_shots is not None:
        summary_df = summary_df[summary_df["few_shots"] == few_shots].copy()

    if summary_df.empty:
        logger.warning("No summary rows found after filtering few_shots=%s; skipping plots.", few_shots)
        return []

    plot_df: pd.DataFrame = summary_df.sort_values(["model_idx", "cot"], ascending=[True, True])
    plot_df["model_size_b"] = plot_df["model_name"].apply(
        lambda n: _extract_model_size(str(n)) if n is not None else None
    )

    if plot_df["model_size_b"].isna().any():
        missing_models = plot_df[plot_df["model_size_b"].isna()]["model_name"].dropna().unique().tolist()
        logger.warning(
            "Could not extract model size for %d rows. Models: %s",
            int(plot_df["model_size_b"].isna().sum()),
            missing_models,
        )
        plot_df.loc[plot_df["model_size_b"].isna(), "model_size_b"] = plot_df.loc[
            plot_df["model_size_b"].isna(), "model_idx"
        ].astype(float)

    plot_df = plot_df.sort_values(["model_size_b", "cot"], ascending=[True, True])

    x_vals = sorted({float(x) for x in plot_df["model_size_b"].unique().tolist()})

    df_no = plot_df[plot_df["cot"] == False].sort_values("model_size_b")
    df_yes = plot_df[plot_df["cot"] == True].sort_values("model_size_b")

    color_no_cot = "blue"
    color_cot = "red"

    x_no = df_no["model_size_b"].astype(float).tolist()
    x_yes = df_yes["model_size_b"].astype(float).tolist()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 4), sharex=True)

    metric_specs = [
        ("accuracy", "Accuracy", axes[0]),
        ("average_token_usage", "Avg. token usage", axes[1]),
        ("average_elapsed_time", "Avg. time (s)", axes[2]),
    ]

    legend_handles = None
    legend_labels = None

    for metric_col, y_label, ax in metric_specs:
        y_no = df_no[metric_col].astype(float).tolist()
        y_yes = df_yes[metric_col].astype(float).tolist()

        h1 = ax.scatter(x_no, y_no, color=color_no_cot, label="No CoT", s=50)
        h2 = ax.scatter(x_yes, y_yes, color=color_cot, label="CoT", s=50)

        ax.set_ylabel(y_label)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{v:g}" for v in x_vals])
        ax.grid(True, axis="y", alpha=0.3)

        if legend_handles is None:
            legend_handles = [h1, h2]
            legend_labels = ["No CoT", "CoT"]

    for ax in axes:
        ax.set_xlabel("Model size (B)")

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.92))

    suffix = f"_shot{few_shots}" if few_shots is not None else ""
    out_path = out_dir / f"multi_agent_model_size_summary{suffix}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    model_cols = sorted(int(v) for v in plot_df["model_idx"].unique().tolist())

    n_cols = len(model_cols)
    fig_cm, axes_cm = plt.subplots(nrows=2, ncols=n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes_cm = np.array(axes_cm).reshape(2, 1)

    cot_rows = [False, True]
    for row_i, cot_val in enumerate(cot_rows):
        for col_i in range(n_cols):
            ax = axes_cm[row_i, col_i]

            model_idx_val = model_cols[col_i]
            subset_df = plot_df[(plot_df["model_idx"] == model_idx_val) & (plot_df["cot"] == cot_val)]

            true_labels = []
            predicted_labels = []
            for _, row in subset_df.iterrows():
                results_file = results_dir / f"multi_agent_model{int(row['model_idx'])}_CoT{row['cot']}_Shot{int(row['few_shots'])}.json"
                if results_file.exists():
                    with results_file.open("r", encoding="utf-8") as f:
                        results = json.load(f)
                    for r in results:
                        true_labels.append(r.get("true_label", ""))
                        predicted_labels.append(r.get("predicted_label", ""))

            if true_labels and predicted_labels:
                cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
            else:
                cm = np.zeros((len(labels), len(labels)), dtype=int)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(ax=ax, cmap="Blues", colorbar=False)

            if row_i == 0:
                ax.set_title(f"Model {model_idx_val}")
            else:
                ax.set_title("")

            if col_i == 0:
                ax.set_ylabel(f"{'CoT' if cot_val else 'No CoT'}\nTrue label")
            else:
                ax.set_ylabel("")

            if row_i == len(cot_rows) - 1:
                ax.set_xlabel("Predicted label")
            else:
                ax.set_xlabel("")

    fig_cm.tight_layout()
    out_path_cm = out_dir / f"multi_agent_model_size_confusion_matrix{suffix}.png"
    fig_cm.savefig(out_path_cm, dpi=200)
    plt.close(fig_cm)
    print(f"Saved multi-agent model size summary plots to: {out_dir}")


def plot_prompt_summary(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    model_idx: Optional[list[int]] = None,
):
    results_dir = results_dir or Path(RESULTS_DIR)
    out_dir = out_dir or Path(FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df is None or df.empty:
        logger.warning("No summary data provided to plot_prompt_summary; skipping plots.")
        return

    required_cols = {
        "model_idx",
        "model_name",
        "cot",
        "few_shots",
        "accuracy",
        "average_token_usage",
        "average_elapsed_time",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"plot_prompt_summary missing columns: {sorted(missing)}")

    plot_df = df.copy()
    if model_idx is not None:
        plot_df = plot_df[plot_df["model_idx"].isin(model_idx)].copy()
        if plot_df.empty:
            logger.warning("No rows found for model_idx=%s; skipping plot.", model_idx)
            return
    plot_df["few_shots"] = plot_df["few_shots"].astype(int)
    plot_df = plot_df.sort_values(["model_idx", "cot", "few_shots"], ascending=[True, True, True])

    model_order = (
        plot_df[["model_idx", "model_name"]]
        .drop_duplicates(subset=["model_idx"])
        .sort_values("model_idx")
    )
    model_idx_to_label = {}
    for idx, name in zip(model_order["model_idx"].tolist(), model_order["model_name"].tolist(), strict=False):
        size_b = _extract_model_size(str(name)) if name is not None else None
        model_idx_to_label[int(idx)] = f"{size_b:g}B" if size_b is not None else (str(name) if name else f"model_{idx}")

    base_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    model_colors = {
        int(idx): base_colors[i % len(base_colors)]
        for i, idx in enumerate(model_order["model_idx"].tolist())
    }
    cot_linestyle = {False: "-", True: "--"}
    cot_marker = {False: "o", True: "s"}

    x_vals = sorted({int(x) for x in plot_df["few_shots"].unique().tolist()})

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 4), sharex=True)

    metric_specs = [
        ("accuracy", "Accuracy", axes[0]),
        ("average_token_usage", "Avg. token usage", axes[1]),
        ("average_elapsed_time", "Avg. time (s)", axes[2]),
    ]

    handles = []
    labels = []

    for (model_idx_val, cot_val), group in plot_df.groupby(["model_idx", "cot"], sort=True):
        model_idx_val = int(model_idx_val)
        cot_val = bool(cot_val)

        by_shots = {int(r["few_shots"]): r for _, r in group.iterrows()}
        line_label = f"{model_idx_to_label.get(model_idx_val, f'model_{model_idx_val}')} ({'CoT' if cot_val else 'No CoT'})"
        color = model_colors.get(model_idx_val, "C0")
        linestyle = cot_linestyle[cot_val]
        marker = cot_marker[cot_val]

        legend_handle = None
        for metric_col, y_label, ax in metric_specs:
            y = [float(by_shots[s][metric_col]) if s in by_shots else float("nan") for s in x_vals]
            (line,) = ax.plot(
                x_vals,
                y,
                marker=marker,
                linewidth=2,
                color=color,
                linestyle=linestyle,
                label=line_label,
            )
            if cot_val:
                line.set_dashes((8, 4))
            if legend_handle is None:
                legend_handle = line
            ax.set_ylabel(y_label)
            ax.grid(True, axis="y", alpha=0.3)

        if legend_handle is not None:
            handles.append(legend_handle)
        labels.append(line_label)

    for ax in axes:
        ax.set_xlabel("# few-shots")
        ax.set_xticks(x_vals)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=3.0,
        markerscale=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_path = out_dir / "multi_agent_prompt_analysis_by_shots.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    cm_labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    cot_rows = [False, True]
    shot_values = sorted({int(x) for x in plot_df["few_shots"].unique().tolist()})

    for model_idx_val in sorted(plot_df["model_idx"].unique()):
        model_idx_val = int(model_idx_val)
        n_cols = len(shot_values)
        if n_cols == 0:
            continue

        fig_cm, axes_cm = plt.subplots(nrows=2, ncols=n_cols, figsize=(4 * n_cols, 8))
        if n_cols == 1:
            axes_cm = np.array(axes_cm).reshape(2, 1)

        for row_i, cot_val in enumerate(cot_rows):
            for col_i, shots in enumerate(shot_values):
                ax = axes_cm[row_i, col_i]

                results_file = results_dir / f"multi_agent_model{model_idx_val}_CoT{cot_val}_Shot{shots}.json"
                if results_file.exists():
                    with results_file.open("r", encoding="utf-8") as f:
                        results = json.load(f)
                    true_labels = [r.get("true_label", "") for r in results]
                    predicted_labels = [r.get("predicted_label", "") for r in results]
                    cm = confusion_matrix(true_labels, predicted_labels, labels=cm_labels)
                else:
                    cm = np.zeros((len(cm_labels), len(cm_labels)), dtype=int)

                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
                disp.plot(ax=ax, cmap="Blues", colorbar=False)

                if col_i == 0:
                    ax.set_ylabel(f"{'CoT' if cot_val else 'No CoT'}\nTrue label")
                else:
                    ax.set_ylabel("")
                if row_i == len(cot_rows) - 1:
                    ax.set_xlabel("Predicted label")
                else:
                    ax.set_xlabel("")

                if row_i == 0:
                    ax.set_title(f"{shots}-shot")
                else:
                    ax.set_title("")

        fig_cm.tight_layout()
        out_path_cm = out_dir / f"multi_agent_prompt_confusion_matrix_model{model_idx_val}.png"
        fig_cm.savefig(out_path_cm, dpi=200)
        plt.close(fig_cm)

    print(f"Saved multi-agent prompt analysis figure to: {out_path}")


def plot_single_run_confusion_matrix(
    results: List[Dict[str, str | int | float]],
    model_idx: int,
    cot: bool,
    few_shots: int,
    out_dir: Path | None = None,
) -> Optional[Path]:
    out_dir = out_dir or Path(FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    true_labels = [str(r.get("true_label", "")) for r in results]
    predicted_labels = [str(r.get("predicted_label", "")) for r in results]

    if not true_labels or not predicted_labels:
        logger.warning("No predictions available to plot single confusion matrix.")
        return None

    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("")

    out_path_cm = out_dir / f"multi_agent_single_confusion_matrix_model{model_idx}_CoT{cot}_shot{few_shots}.png"
    fig_cm.tight_layout()
    fig_cm.savefig(out_path_cm, dpi=200)
    plt.close(fig_cm)
    print(f"Saved single-run confusion matrix to: {out_path_cm}")
    return out_path_cm


def model_size_analysis():
    _, test_df = load_data_split(dataset_key=DATASET_KEY)
    n_few_shot = [0]
    cot = config["prompting"]["cot"]
    params_comb = [(cot_val, few_shot_val) for cot_val in cot for few_shot_val in n_few_shot]
    model_indices = list(range(len(config["models"])))
    for model_idx in model_indices:
        primary_model = _build_model(model_idx)
        critique_model = _build_model(model_idx)
        for cot_val, few_shot_val in params_comb:
            print(f"Evaluating multi-agent model {model_idx} with cot={cot_val} and few_shots={few_shot_val}...")
            evaluate_model_on_test_set(test_df, primary_model, critique_model, model_idx, cot=cot_val, few_shots=few_shot_val)

    df = summarize_results()
    summary_path = save_summary_to_file(df=df)
    print(f"Wrote multi-agent results summary CSV to: {summary_path}")
    plot_model_size_summary(df=df)
    print("Saved multi-agent model size analysis plots.")


def prompt_analysis():
    _, test_df = load_data_split(dataset_key=DATASET_KEY)
    n_few_shot = config["prompting"]["n_few_shots"]
    cot = config["prompting"]["cot"]
    params_comb = [(cot_val, few_shot_val) for cot_val in cot for few_shot_val in n_few_shot]
    model_indices = [2, 3]
    for model_idx in model_indices:
        primary_model = _build_model(model_idx)
        critique_model = _build_model(model_idx)
        for cot_val, few_shot_val in params_comb:
            print(f"Evaluating multi-agent model {model_idx} with cot={cot_val} and few_shots={few_shot_val}...")
            evaluate_model_on_test_set(test_df, primary_model, critique_model, model_idx, cot=cot_val, few_shots=few_shot_val)

    df = summarize_results()
    summary_path = save_summary_to_file(df=df)
    print(f"Wrote multi-agent results summary CSV to: {summary_path}")
    plot_prompt_summary(df=df, model_idx=model_indices)
    print("Saved multi-agent prompt analysis plots.")


def main(mode: Optional[str] = None) -> None:
    if mode is not None:
        mode = mode.strip().lower()
    else:
        mode = (sys.argv[1].strip().lower() if len(sys.argv) > 1 else "model_size")
    if mode in {"model_size", "model", "size"}:
        model_size_analysis()
    elif mode in {"prompt", "prompting", "shots"}:
        prompt_analysis()
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'model_size' or 'prompt'.")


def run_best_model(model_idx: int, cot: bool, few_shots: int) -> None:
    _, test_df = load_data_split(dataset_key=DATASET_KEY)
    primary_model = _build_model(model_idx)
    critique_model = _build_model(model_idx)
    print(f"Evaluating best multi-agent model {model_idx} with cot={cot} and few_shots={few_shots}...")
    results = evaluate_model_on_test_set(test_df, primary_model, critique_model, model_idx, cot=cot, few_shots=few_shots)
    accuracy, macro_f1, adjacent_accuracy, avg_token_usage, avg_elapsed_time = _compute_metrics(results)
    best_summary = {
        "method": "multi_agent",
        "dataset_key": DATASET_KEY,
        "model_idx": model_idx,
        "model_name": str(config.get("models", [""])[model_idx]) if model_idx < len(config.get("models", [])) else "",
        "cot": bool(cot),
        "few_shots": int(few_shots),
        "n_samples": len(results),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "adjacent_accuracy": adjacent_accuracy,
        "average_token_usage": avg_token_usage,
        "average_elapsed_time": avg_elapsed_time,
    }
    best_summary_path = Path(RESULTS_DIR) / f"multi_agent_best_model{model_idx}_CoT{cot}_Shot{few_shots}_summary.json"
    with best_summary_path.open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)

    print(
        "Best multi-agent model "
        f"{model_idx} results - Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, "
        f"Adjacent accuracy: {adjacent_accuracy:.4f}, Avg. token usage: {avg_token_usage:.2f}, "
        f"Avg. time: {avg_elapsed_time:.2f}s"
    )
    print(f"Saved best-run summary JSON to: {best_summary_path}")
    plot_single_run_confusion_matrix(results, model_idx=model_idx, cot=cot, few_shots=few_shots)


if __name__ == "__main__":
    # Run the best model with chosen parameters (use the same parameters for the two datasets for consistency)
    best_model_idx = 2
    best_model_cot = True
    best_model_few_shots = 4
    
    run_best_model(model_idx=best_model_idx, cot=best_model_cot, few_shots=best_model_few_shots)

    # To summarize results from previous runs without re-running analyses, uncomment below:
    # df = summarize_results()
    # save_summary_to_file(df=df)
    # print(f"Summarized results: \n{df}")

    # Batch run analyses (uncomment to run full analyses):
    # main(mode="model_size")
    # main(mode="prompt")
