from __future__ import annotations

import sys
sys.path.append("..")  # Add parent directory to path for imports

import logging
from typing import Dict, List, Tuple
import time
import json
import yaml
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from pathlib import Path

from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from data_loader import load_data_split
from prompt import SYSTEM_PROMPT_BASE, SYSTEM_PROMPT_COT, system_prompt_few_shots
from config import RESULTS_DIR

logger = logging.getLogger(__name__)

config = yaml.safe_load(open("config.yaml", "r"))
if not config.get("api_key") or not config.get("api_key").strip(): # None or empty string
    logger.warning("No API key found in config.yaml; will attempt to retrieve from inference service")
    from utils.inference_auth_token import get_access_token # for inference service
    config["api_key"] = get_access_token()
    config["model_provider"] = "openai" # openai-compatible API




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


def evaluate_model_on_test_set(test_df: pd.DataFrame, model: BaseChatModel, cot: bool, few_shots: int) -> List[Dict[str, str | int | float]]:
    """
    Evaluate the LLM on the test set and return a list of results containing the input text, true label, predicted label, 
    rationale, token usage, and elapsed time for each test sample.
    Addtionally, the results are saved to a JSON file in the RESULTS_DIR for later analysis. The filename is "llm_<cot>_<few_shots>.json".
    Args:
        test_df: A DataFrame containing the test samples with columns "text" and "cefr_level".
        model: The LLM instance to use for evaluation.
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
    out_path = results_dir / f"llm_{cot}_{few_shots}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def main():
    train_df, test_df = load_data_split()
    n_few_shot = config["prompting"]["n_few_shots"]
    cot = config["prompting"]["cot"]
    params_comb = [(cot_val, few_shot_val) for cot_val in cot for few_shot_val in n_few_shot]
    model_indices = list(range(len(config["models"])))
    for model_idx in model_indices:
        model = _build_model(model_idx)
        for cot_val, few_shot_val in params_comb:
            print(f"Evaluating model with cot={cot_val} and few_shots={few_shot_val}...")
            evaluate_model_on_test_set(test_df, model, cot=cot_val, few_shots=few_shot_val)
    # Further analysis of results can be done here (e.g., calculating accuracy, analyzing rationales, etc.)



if __name__ == "__main__":
    main()
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