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

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from data_loader import load_data_split
from prompt import SYSTEM_PROMPT_BASE, SYSTEM_PROMPT_COT, system_prompt_few_shots
from config import RESULTS_DIR, FIGURES_DIR

logger = logging.getLogger(__name__)

if not config.get("api_key") or not config.get("api_key").strip(): # None or empty string
    logger.warning("No API key found in config.yaml; will attempt to retrieve from inference service")
    from utils.inference_auth_token import get_access_token # for inference service
    config["api_key"] = get_access_token()
    config["model_provider"] = "openai" # openai-compatible API


def _extract_model_size(model_name: str) -> Optional[float]:
    """
    Extract the model size in billions of parameters from the model name string. 
    This function looks for patterns like "70B", "3B", "350M" in the model name and converts them to a float representing billions of parameters.
    
    Args:
        model_name: The name of the model, which may contain a size indicator (e.g., "meta-llama/Meta-Llama-3.1-70B-Instruct").
    Returns:
        The extracted model size in billions of parameters as a float (e.g., 70.0 for "70B"), or None if no size pattern is found.
    """
    size_pattern = re.compile(r"(\d+(?:\.\d+)?)([MB])", re.IGNORECASE)
    match = size_pattern.search(model_name)
    if match:
        number_str, unit = match.groups()
        try:
            number = float(number_str)
            if unit.upper() == "M":
                return number / 1000  # Convert millions to billions
            elif unit.upper() == "B":
                return number  # Already in billions
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


def _message_text(message: AIMessage | HumanMessage | SystemMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def _extract_final_answer(answer: str) -> Tuple[str, str]:
    """
    Extract the final answer and rationale from the model's response text. The final answer is expected to follow a "#### " delimiter.
    
    Args: 
        the raw text response from the model, which may contain a rationale and a final answer separated by "#### ".
    Returns:
        a tuple of (final_answer, rationale). The final answer is the text after "#### ", and the rationale is the text before it. 
        If "#### " is not found, the entire text is treated as the final answer and the rationale is empty.
    """
    # print(f"Raw model response:\n{answer}\n")
    # print("=" * 50)
    start_tag = "#### "
    start_idx = answer.find(start_tag)
    if start_idx != -1:
        rationale = answer[:start_idx].strip()
        answer = answer[start_idx + len(start_tag):].strip()
    else: # fall back to try extracting the final answer from the last line
        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        last_line = answer.strip().splitlines()[-1].strip()
        for level in levels:
            if level in last_line:
                rationale = answer.strip()
                answer = level
                break
        else:
            rationale = answer.strip()
            answer = "Unclear"
    return answer, rationale


def _record_usage(response: AIMessage, include: bool = True) -> int:
    """
    Record the token usage from the model's response. This function checks for various fields in the response's usage metadata to calculate total tokens used.
    
    Args:
        response: The raw response from the model, which may contain usage metadata.
        include: Whether to include this response's token usage in the total count. Defaults to True.
    Returns:
        The updated total token usage count after including this response's usage.
    """
    if not include or response is None:
        return 0
    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    total = 0
    if input_tokens is not None:
        total += int(input_tokens)
    if output_tokens is not None:
        total += int(output_tokens)
    if not total:
        fallback = usage.get("total_tokens") or usage.get("token_count")
        if fallback is not None:
            total = int(fallback)
    return total


def llm(model: BaseChatModel, text: str, cot: bool = False, few_shots: int = 0) -> tuple[str, str, int, float]:
    """Run a single-turn hypothesis improvement using only an LLM.
    Args:
        model: The LLM instance to use for inference.
        text: The input text to rate the CEFR level of.
        cot: Whether to include a chain-of-thought rationale in the prompt. Defaults to False.
        few_shots: The number of few-shot examples to include in the prompt. Defaults to 0.
    Returns:
        - The final CEFR level predicted by the model in string labels (e.g., "A1", "B2", etc.).
        - The rationale for the prediction provided by the model.
        - The total token usage during the workflow run.
        - The total time taken for the workflow run.
    """
    now = time.time()
    text = text.strip()
    if not text:
        raise ValueError("Input text is empty.")

    if few_shots > 0:
        system_prompt = system_prompt_few_shots(n=few_shots, cot=cot)
    else:
        system_prompt = SYSTEM_PROMPT_COT if cot else SYSTEM_PROMPT_BASE

    # print(f"System prompt:\n{system_prompt}\n")
    # print("=" * 50)

    token_usage = 0

    conversation: List[HumanMessage | AIMessage | SystemMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Text:\n{text}"),
    ]

    assistant_message = model.invoke(conversation)
    token_usage += _record_usage(assistant_message)
    assistant_text = _message_text(assistant_message)

    answer, rationale = _extract_final_answer(assistant_text)
    return answer, rationale, token_usage, time.time() - now


def evaluate_model_on_test_set(test_df: pd.DataFrame, model: BaseChatModel, model_idx: int, cot: bool, few_shots: int) -> List[Dict[str, str | int | float]]:
    """
    Evaluate the LLM on the test set and return a list of results containing the input text, true label, predicted label, 
    rationale, token usage, and elapsed time for each test sample.
    Addtionally, the results are saved to a JSON file in the RESULTS_DIR for later analysis. The filename is "llm_model{model_idx}_CoT{cot}_Shot{few_shots}.json".
    Args:
        test_df: A DataFrame containing the test samples with columns "text" and "cefr_level".
        model: The LLM instance to use for evaluation.
        model_idx: The index of the model being evaluated (for identification purposes).
        cot: Whether to include a chain-of-thought rationale in the prompt.
        few_shots: The number of few-shot examples to include in the prompt.
    Returns: A list of dictionaries, where each dictionary contains the following keys:
        - "text": The input text that was evaluated.
        - "true_label": The true CEFR level of the input text.
        - "predicted_label": The CEFR level predicted by the model.
        - "rationale": The rationale for the predicted CEFR level.
        - "token_usage": The total token usage during the evaluation.
        - "elapsed_time": The total time taken for the evaluation.
    """
    parallel_cfg = config.get("parallelism", {}) if isinstance(config, dict) else {}
    max_workers = int(parallel_cfg.get("max_workers", 1) or 1)

    def _eval_one(i: int) -> Dict[str, str | int | float]:
        row = test_df.iloc[i]
        text = str(row["text"])
        true_label = row["cefr_level"]
        predicted_label, rationale, token_usage, elapsed_time = llm(model, text, cot=cot, few_shots=few_shots)
        return {
            "text": text,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "rationale": rationale,
            "token_usage": token_usage,
            "elapsed_time": elapsed_time,
        }

    results_by_idx: Dict[int, Dict[str, str | int | float]] = {}
    if max_workers <= 1:
        for i in tqdm.tqdm(range(len(test_df)), desc="Evaluating"):
            results_by_idx[i] = _eval_one(i)
    else:
        logger.info("Evaluating %d samples with max_workers=%d", len(test_df), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_eval_one, i): i for i in range(len(test_df))}
            for fut in tqdm.tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="Evaluating",
            ):
                i = future_to_idx[fut]
                results_by_idx[i] = fut.result()

    results = [results_by_idx[i] for i in range(len(test_df))]

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"llm_model{model_idx}_CoT{cot}_Shot{few_shots}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results



_RUN_FILE_RE = re.compile(
    r"^llm_model(?P<model_idx>\d+)_CoT(?P<cot>True|False)_Shot(?P<shots>\d+)\.json$"
)


def _compute_metrics(results: List[Dict[str, str | int | float]]) -> Tuple[float, float, float]:
    total_samples = len(results)
    if total_samples == 0:
        return 0.0, 0.0, 0.0
    correct_predictions = sum(
        1
        for r in results
        if str(r.get("true_label")) == str(r.get("predicted_label"))
    )
    accuracy = correct_predictions / total_samples
    avg_token_usage = (
        sum(float(r.get("token_usage", 0) or 0) for r in results) / total_samples
    )
    avg_elapsed_time = (
        sum(float(r.get("elapsed_time", 0) or 0) for r in results) / total_samples
    )
    return accuracy, avg_token_usage, avg_elapsed_time


def summarize_results(results_dir: Path | None = None) -> pd.DataFrame:
    """
    Summarize evaluation results per JSON file (i.e., per model_idx x cot x few_shots combination).
    Reads all JSON result files in RESULTS_DIR matching the pattern "llm_model*_CoT*_Shot*.json".
    
    Returns:
        A DataFrame with columns: model_idx, model_name, cot, few_shots, accuracy, average_token_usage, average_elapsed_time.
    """
    results_dir = results_dir or Path(RESULTS_DIR)
    rows: List[Dict[str, str | int | float | bool]] = []

    for file in sorted(results_dir.glob("llm_model*_CoT*_Shot*.json")):
        match = _RUN_FILE_RE.match(file.name)
        if not match:
            continue

        model_idx = int(match.group("model_idx"))
        cot = match.group("cot") == "True"
        few_shots = int(match.group("shots"))

        with file.open("r", encoding="utf-8") as f:
            results = json.load(f)

        accuracy, avg_token_usage, avg_elapsed_time = _compute_metrics(results)
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
            "average_token_usage",
            "average_elapsed_time",
        ],
    )


def save_summary_to_file(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Write a CSV summary (one row per model/cot/shots JSON file) to disk and create plots to visualize."""
    results_dir = results_dir or Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (results_dir / "llm_results_summary.csv")
    df.to_csv(out_path, index=False)
    return out_path


def plot_model_size_summary(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    few_shots: Optional[int] = None,
) -> List[Path]:
    """Create 3 figures (accuracy, avg token usage, avg elapsed time) split by CoT.

    X-axis: model name
    Y-axis: metric value
    Per model: two datapoints (CoT=True in red, CoT=False in blue)

    If multiple `few_shots` values exist in the summary, pass `few_shots` to select one.

    Returns:
        List of paths to the saved figures.
    """

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
            # Default to 0-shot if present; otherwise take the smallest.
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
        # Fallback: use model_idx as a pseudo-x value to avoid crashing.
        plot_df.loc[plot_df["model_size_b"].isna(), "model_size_b"] = plot_df.loc[
            plot_df["model_size_b"].isna(), "model_idx"
        ].astype(float)

    # Order by size (ascending)
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
    out_path = out_dir / f"llm_model_size_summary{suffix}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return [out_path]


def plot_prompt_summary(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    model_idx: Optional[list[int]] = None,
):
    """Plot prompt-analysis results across number of shots.

    X-axis: few_shots
    Y-axis: (accuracy, average_token_usage, average_elapsed_time)
    Lines: one per (model_idx, cot) => 2 models x 2 cot = 4 lines (for prompt_analysis())

    Saves one figure with 3 subplots and a shared legend.
    """
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

    # Determine model labels (prefer names; fallback to model_{idx}).
    model_order = (
        plot_df[["model_idx", "model_name"]]
        .drop_duplicates(subset=["model_idx"])
        .sort_values("model_idx")
    )
    model_idx_to_label = {}
    for idx, name in zip(model_order["model_idx"].tolist(), model_order["model_name"].tolist(), strict=False):
        size_b = _extract_model_size(str(name)) if name is not None else None
        model_idx_to_label[int(idx)] = f"{size_b:g}B" if size_b is not None else (str(name) if name else f"model_{idx}")

    # Styles: color by model, line style by CoT.
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

        # Align points to the full x grid.
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
            # Make dashed lines visually more distinct than default "--".
            if cot_val:
                line.set_dashes((8, 4))
            if legend_handle is None:
                legend_handle = line
            ax.set_ylabel(y_label)
            ax.grid(True, axis="y", alpha=0.3)

        # Collect legend handles once.
        if legend_handle is not None:
            handles.append(legend_handle)
        labels.append(line_label)

    for ax in axes:
        ax.set_xlabel("# few-shots")
        ax.set_xticks(x_vals)

    # Shared legend across subplots.
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

    out_path = out_dir / "llm_prompt_analysis_by_shots.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved prompt analysis figure to: {out_path}")


def model_size_analysis():
    _, test_df = load_data_split()
    n_few_shot = [0] # only do 0-shot for model size analysis to isolate the effect of model size without few-shot examples
    cot = config["prompting"]["cot"]
    params_comb = [(cot_val, few_shot_val) for cot_val in cot for few_shot_val in n_few_shot]
    model_indices = list(range(len(config["models"])))
    for model_idx in model_indices:
        model = _build_model(model_idx)
        for cot_val, few_shot_val in params_comb:
            print(f"Evaluating model {model_idx} with cot={cot_val} and few_shots={few_shot_val}...")
            evaluate_model_on_test_set(test_df, model, model_idx, cot=cot_val, few_shots=few_shot_val)

    df = summarize_results()
    summary_path = save_summary_to_file(df=df)
    print(f"Wrote results summary CSV to: {summary_path}")
    plot_model_size_summary(df=df)
    print("Saved model size analysis plots.")


def prompt_analysis():
    _, test_df = load_data_split()
    n_few_shot = config["prompting"]["n_few_shots"]
    cot = config["prompting"]["cot"]
    params_comb = [(cot_val, few_shot_val) for cot_val in cot for few_shot_val in n_few_shot]
    model_indices = [2, 3] # only evaluate the two best-performing models from the model size analysis to reduce the number of runs needed for prompt analysis
    for model_idx in model_indices:
        model = _build_model(model_idx)
        for cot_val, few_shot_val in params_comb:
            print(f"Evaluating model {model_idx} with cot={cot_val} and few_shots={few_shot_val}...")
            evaluate_model_on_test_set(test_df, model, model_idx, cot=cot_val, few_shots=few_shot_val)

    df = summarize_results()
    summary_path = save_summary_to_file(df=df)
    print(f"Wrote results summary CSV to: {summary_path}")
    plot_prompt_summary(df=df, model_idx=model_indices)
    print("Saved prompt analysis plots.")


def main(mode: Optional[str] = None) -> None:
    """Simple entrypoint.

    Usage:
        python llm.py model_size
        python llm.py prompt

    Defaults to `model_size` if no argument is provided.
    """
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


if __name__ == "__main__":
    # main(mode="model_size")
    # main(mode="prompt")
    # plot_model_size_summary(df=summarize_results())
    plot_prompt_summary(df=summarize_results(), model_idx=[2, 3])
    
    # model_idx = 0
    # cot = True
    # few_shots = 0

    # Load the dataset
    # train_df, test_df = load_data_split()

    # Single-run example on the first test sample
    # text = test_df["text"][0]
    # print(f"Input text:\n{text}\n")

    # model = _build_model(model_idx)
    # answer, rationale, token_usage, elapsed_time = llm(model, text, cot=cot, few_shots=few_shots)
    # if answer is not None:
    #     print(f"Predicted CEFR level (integer label): {answer}")
    #     print("\nRationale:\n")
    #     print(rationale)
    # print(f"\nApproximate tokens used: {token_usage}")
    # print(f"Total time taken for the workflow run: {elapsed_time:.2f} seconds")

    # Run evaluation on the entire test set
    # model = _build_model(model_idx)
    # evaluate_model_on_test_set(test_df, model, cot=cot, few_shots=few_shots)