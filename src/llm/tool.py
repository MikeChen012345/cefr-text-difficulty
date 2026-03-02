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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from data_loader import load_data_split
from prompt import build_system_prompt_tool
from config import results_dir, figures_dir

logger = logging.getLogger(__name__)

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

if not config.get("api_key") or not config.get("api_key").strip():
    logger.warning("No API key found in config.yaml; will attempt to retrieve from inference service")
    from utils.inference_auth_token import get_access_token
    config["api_key"] = get_access_token()
    config["model_provider"] = "openai"

DATASET_KEY = "readme_en" # "cefr_sp_en" # 
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


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s and s.strip()]


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


@tool
def analyze_vocabulary(text: str) -> Dict[str, float | int]:
    """Analyze vocabulary range and complexity in the input text."""
    words = _tokenize_words(text)
    if not words:
        return {
            "word_count": 0,
            "unique_word_count": 0,
            "type_token_ratio": 0.0,
            "avg_word_length": 0.0,
            "long_word_ratio": 0.0,
        }

    unique_words = set(words)
    long_words = [w for w in words if len(w) >= 7]
    return {
        "word_count": len(words),
        "unique_word_count": len(unique_words),
        "type_token_ratio": round(len(unique_words) / len(words), 4),
        "avg_word_length": round(float(np.mean([len(w) for w in words])), 3),
        "long_word_ratio": round(len(long_words) / len(words), 4),
    }


@tool
def analyze_grammar(text: str) -> Dict[str, float | int]:
    """Estimate grammatical variety/accuracy using lightweight heuristics."""
    sentences = _split_sentences(text)
    words = _tokenize_words(text)
    if not sentences:
        return {
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "capitalization_issues": 0,
            "punctuation_issues": 0,
        }

    sentence_word_counts = [len(_tokenize_words(s)) for s in sentences]
    capitalization_issues = sum(1 for s in sentences if s and not s[0].isupper())
    punctuation_issues = len(re.findall(r"[!?.,]{2,}", text))
    if text.strip() and text.strip()[-1] not in ".!?":
        punctuation_issues += 1

    return {
        "sentence_count": len(sentences),
        "avg_sentence_length": round(float(np.mean(sentence_word_counts)), 3),
        "sentence_length_std": round(float(np.std(sentence_word_counts)), 3),
        "capitalization_issues": capitalization_issues,
        "punctuation_issues": punctuation_issues,
        "approx_clause_markers": sum(words.count(x) for x in ["that", "which", "who", "because", "although", "while"]),
    }


@tool
def analyze_cohesion(text: str) -> Dict[str, float | int]:
    """Analyze cohesion/coherence signals (connectors and local lexical overlap)."""
    sentences = _split_sentences(text)
    words = _tokenize_words(text)
    markers = {
        "however", "therefore", "moreover", "furthermore", "first", "second", "finally",
        "because", "although", "while", "then", "thus", "for example", "in addition",
    }

    marker_count = sum(1 for w in words if w in markers)
    pronoun_count = sum(1 for w in words if w in {"he", "she", "it", "they", "this", "that", "these", "those"})

    overlaps: List[float] = []
    for i in range(len(sentences) - 1):
        a = set(_tokenize_words(sentences[i]))
        b = set(_tokenize_words(sentences[i + 1]))
        union = len(a | b)
        overlaps.append((len(a & b) / union) if union else 0.0)

    return {
        "sentence_count": len(sentences),
        "discourse_marker_count": marker_count,
        "discourse_marker_density": round(marker_count / max(1, len(sentences)), 4),
        "adjacent_sentence_overlap": round(float(np.mean(overlaps)) if overlaps else 0.0, 4),
        "pronoun_count": pronoun_count,
    }


@tool
def analyze_task_achievement(text: str) -> Dict[str, float | int | str]:
    """Approximate task achievement: completeness, organization, and register signals."""
    sentences = _split_sentences(text)
    words = _tokenize_words(text)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    contractions = re.findall(r"\b\w+'(t|re|ve|ll|d|m|s)\b", text.lower()) # match can't, we're, I've, I'll etc.
    avg_sentence_len = float(np.mean([len(_tokenize_words(s)) for s in sentences])) if sentences else 0.0

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_sentence_length": round(avg_sentence_len, 3),
        "contraction_count": len(contractions),
        "length_band": "short" if len(words) < 60 else ("medium" if len(words) < 180 else "long"),
    }


@tool
def compose_cefr_features(text: str) -> Dict[str, Dict[str, float | int]]:
    """Return all CEFR-oriented feature analyses together in one call."""
    return {
        "vocabulary": analyze_vocabulary.invoke({"text": text}),
        "grammar": analyze_grammar.invoke({"text": text}),
        "cohesion": analyze_cohesion.invoke({"text": text}),
        "task_achievement": analyze_task_achievement.invoke({"text": text}),
    }


TOOLS = [
    analyze_vocabulary,
    analyze_grammar,
    analyze_cohesion,
    analyze_task_achievement,
    compose_cefr_features,
]
TOOL_REGISTRY = {tool_obj.name: tool_obj for tool_obj in TOOLS}


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


def _invoke_with_tools(model: BaseChatModel, system_prompt: str, text: str) -> tuple[str, int, Dict[str, int]]:
    tool_model = model.bind_tools(TOOLS)
    messages: List[HumanMessage | AIMessage | SystemMessage | ToolMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Text:\n{text}\n\n"),
    ]
    token_usage = 0
    tool_call_counts: Dict[str, int] = {tool_obj.name: 0 for tool_obj in TOOLS}

    for _ in range(6):
        ai_message = tool_model.invoke(messages)
        token_usage += _record_usage(ai_message)
        messages.append(ai_message)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if not tool_calls:
            return _message_text(ai_message), token_usage, tool_call_counts

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            args = tool_call.get("args", {}) or {}
            tool_call_id = tool_call.get("id")
            if tool_name in tool_call_counts:
                tool_call_counts[tool_name] += 1
            if tool_name not in TOOL_REGISTRY:
                tool_output = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    tool_output = TOOL_REGISTRY[tool_name].invoke(args)
                except Exception as exc:
                    tool_output = {"error": str(exc)}
            messages.append(
                ToolMessage(
                    content=json.dumps(tool_output, ensure_ascii=False),
                    tool_call_id=tool_call_id,
                )
            )

    return _message_text(messages[-1]), token_usage, tool_call_counts


def tool_llm(model: BaseChatModel, text: str, cot: bool = False, few_shots: int = 0) -> tuple[str, str, int, float, Dict[str, int]]:
    now = time.time()
    text = text.strip()
    if not text:
        raise ValueError("Input text is empty.")

    system_prompt = build_system_prompt_tool(dataset_key=DATASET_KEY, n=few_shots, cot=cot)

    assistant_text, token_usage, tool_call_counts = _invoke_with_tools(model, system_prompt, text)
    answer, rationale = _extract_final_answer(assistant_text)
    return answer, rationale, token_usage, time.time() - now, tool_call_counts


def evaluate_model_on_test_set(test_df: pd.DataFrame, model: BaseChatModel, model_idx: int, cot: bool, few_shots: int) -> List[Dict[str, object]]:
    parallel_cfg = config.get("parallelism", {}) if isinstance(config, dict) else {}
    max_workers = int(parallel_cfg.get("max_workers", 1) or 1)

    def _eval_one(i: int) -> Dict[str, object]:
        row = test_df.iloc[i]
        text = str(row["text"])
        true_label = row["cefr_level"]
        predicted_label, rationale, token_usage, elapsed_time, tool_call_counts = tool_llm(model, text, cot=cot, few_shots=few_shots)
        return {
            "text": text,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "rationale": rationale,
            "token_usage": token_usage,
            "elapsed_time": elapsed_time,
            "tool_call_counts": tool_call_counts,
            "total_tool_calls": int(sum(tool_call_counts.values())),
        }

    results_by_idx: Dict[int, Dict[str, object]] = {}
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
    out_path = run_results_dir / f"tool_model{model_idx}_CoT{cot}_Shot{few_shots}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


_RUN_FILE_RE = re.compile(
    r"^tool_model(?P<model_idx>\d+)_CoT(?P<cot>True|False)_Shot(?P<shots>\d+)\.json$"
)


def _results_for_metrics(results: List[Dict[str, object]]) -> List[Dict[str, str | int | float]]:
    return [
        {
            "true_label": str(r.get("true_label", "")),
            "predicted_label": str(r.get("predicted_label", "")),
            "token_usage": _safe_float(r.get("token_usage", 0), default=0.0),
            "elapsed_time": _safe_float(r.get("elapsed_time", 0), default=0.0),
        }
        for r in results
    ]


def _safe_float(value: object, default: float = 0.0) -> float:
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


def _average_tool_calls_by_tool(results: List[Dict[str, object]]) -> Dict[str, float]:
    tool_names = [tool_obj.name for tool_obj in TOOLS]
    total_samples = len(results)
    if total_samples == 0:
        return {tool_name: 0.0 for tool_name in tool_names}

    totals: Dict[str, float] = {tool_name: 0.0 for tool_name in tool_names}
    for row in results:
        raw_counts = row.get("tool_call_counts", {})
        counts = raw_counts if isinstance(raw_counts, dict) else {}
        for tool_name in tool_names:
            raw_val = counts.get(tool_name, 0)
            try:
                totals[tool_name] += float(raw_val)
            except (TypeError, ValueError):
                continue

    return {tool_name: totals[tool_name] / total_samples for tool_name in tool_names}


def summarize_results(results_dir: Path | None = None) -> pd.DataFrame:
    results_dir = results_dir or Path(RESULTS_DIR)
    tool_names = [tool_obj.name for tool_obj in TOOLS]
    rows: List[Dict[str, str | int | float | bool]] = []

    for file in sorted(results_dir.glob("tool_model*_CoT*_Shot*.json")):
        match = _RUN_FILE_RE.match(file.name)
        if not match:
            continue

        model_idx = int(match.group("model_idx"))
        cot = match.group("cot") == "True"
        few_shots = int(match.group("shots"))

        with file.open("r", encoding="utf-8") as f:
            results = json.load(f)

        metrics_input = _results_for_metrics(results)
        accuracy, macro_f1, adjacent_accuracy, avg_token_usage, avg_elapsed_time = _compute_metrics(metrics_input)
        avg_tool_calls_by_tool = _average_tool_calls_by_tool(results)
        avg_total_tool_calls = float(sum(avg_tool_calls_by_tool.values()))
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
                "average_total_tool_calls": avg_total_tool_calls,
            }
        )

        for tool_name in tool_names:
            rows[-1][f"average_tool_calls__{tool_name}"] = avg_tool_calls_by_tool.get(tool_name, 0.0)

    columns = [
        "model_idx",
        "model_name",
        "cot",
        "few_shots",
        "accuracy",
        "macro_f1",
        "adjacent_accuracy",
        "average_token_usage",
        "average_elapsed_time",
        "average_total_tool_calls",
    ] + [f"average_tool_calls__{tool_name}" for tool_name in tool_names]

    return pd.DataFrame(rows, columns=columns)


def save_summary_to_file(
    df: pd.DataFrame,
    results_dir: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    results_dir = results_dir or Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (results_dir / "tool_results_summary.csv")
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
    out_path = out_dir / f"tool_model_size_summary{suffix}.png"
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
                results_file = results_dir / f"tool_model{int(row['model_idx'])}_CoT{row['cot']}_Shot{int(row['few_shots'])}.json"
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
    out_path_cm = out_dir / f"tool_model_size_confusion_matrix{suffix}.png"
    fig_cm.savefig(out_path_cm, dpi=200)
    plt.close(fig_cm)
    print(f"Saved tool-enabled model size summary plots to: {out_dir}")


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

    out_path = out_dir / "tool_prompt_analysis_by_shots.png"
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

                results_file = results_dir / f"tool_model{model_idx_val}_CoT{cot_val}_Shot{shots}.json"
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
        out_path_cm = out_dir / f"tool_prompt_confusion_matrix_model{model_idx_val}.png"
        fig_cm.savefig(out_path_cm, dpi=200)
        plt.close(fig_cm)

    print(f"Saved tool-enabled prompt analysis figure to: {out_path}")


def plot_single_run_confusion_matrix(
    results: List[Dict[str, object]],
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

    out_path_cm = out_dir / f"tool_single_confusion_matrix_model{model_idx}_CoT{cot}_shot{few_shots}.png"
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
        model = _build_model(model_idx)
        for cot_val, few_shot_val in params_comb:
            print(f"Evaluating tool-enabled model {model_idx} with cot={cot_val} and few_shots={few_shot_val}...")
            evaluate_model_on_test_set(test_df, model, model_idx, cot=cot_val, few_shots=few_shot_val)

    df = summarize_results()
    summary_path = save_summary_to_file(df=df)
    print(f"Wrote tool-enabled results summary CSV to: {summary_path}")
    plot_model_size_summary(df=df)
    print("Saved tool-enabled model size analysis plots.")


def prompt_analysis():
    _, test_df = load_data_split(dataset_key=DATASET_KEY)
    n_few_shot = config["prompting"]["n_few_shots"]
    cot = config["prompting"]["cot"]
    params_comb = [(cot_val, few_shot_val) for cot_val in cot for few_shot_val in n_few_shot]
    model_indices = [2, 3]
    for model_idx in model_indices:
        model = _build_model(model_idx)
        for cot_val, few_shot_val in params_comb:
            print(f"Evaluating tool-enabled model {model_idx} with cot={cot_val} and few_shots={few_shot_val}...")
            evaluate_model_on_test_set(test_df, model, model_idx, cot=cot_val, few_shots=few_shot_val)

    df = summarize_results()
    summary_path = save_summary_to_file(df=df)
    print(f"Wrote tool-enabled results summary CSV to: {summary_path}")
    plot_prompt_summary(df=df, model_idx=model_indices)
    print("Saved tool-enabled prompt analysis plots.")


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
    model = _build_model(model_idx)
    print(f"Evaluating best tool-enabled model {model_idx} with cot={cot} and few_shots={few_shots}...")
    results = evaluate_model_on_test_set(test_df, model, model_idx, cot=cot, few_shots=few_shots)
    metrics_input = _results_for_metrics(results)
    accuracy, macro_f1, adjacent_accuracy, avg_token_usage, avg_elapsed_time = _compute_metrics(metrics_input)
    avg_tool_calls_by_tool = _average_tool_calls_by_tool(results)
    avg_total_tool_calls = float(sum(avg_tool_calls_by_tool.values()))
    best_summary = {
        "method": "tool",
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
        "average_total_tool_calls": avg_total_tool_calls,
        "average_tool_calls_by_tool": avg_tool_calls_by_tool,
    }
    best_summary_path = Path(RESULTS_DIR) / f"tool_best_model{model_idx}_CoT{cot}_Shot{few_shots}_summary.json"
    with best_summary_path.open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)

    print(
        "Best tool-enabled model "
        f"{model_idx} results - Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}, "
        f"Adjacent accuracy: {adjacent_accuracy:.4f}, Avg. token usage: {avg_token_usage:.2f}, "
        f"Avg. time: {avg_elapsed_time:.2f}s, Avg. tool calls: {avg_total_tool_calls:.2f}"
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
    df = summarize_results()
    save_summary_to_file(df=df)
    print(f"Summarized results: \n{df}")

    # Batch run analyses (uncomment to run full analyses):
    # main(mode="model_size")
    # main(mode="prompt")
