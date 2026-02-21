from __future__ import annotations

import sys
sys.path.append("..")  # Add parent directory to path for imports

import logging
from typing import Dict, List, Tuple
import time
import json
import yaml

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from data_loader import load_data_split
from prompt import SYSTEM_PROMPT_TEMPLATE, SYSTEM_PROMPT_COT, system_prompt_few_shots

logger = logging.getLogger(__name__)

config = yaml.safe_load(open("config.yaml", "r"))
if not config.get("api_key") or not config.get("api_key").strip(): # None or empty string
    logger.warning("No API key found in config.yaml; will attempt to retrieve from inference service")
    from utils.inference_auth_token import get_access_token # for inference service
    config["api_key"] = get_access_token()
    config["model_provider"] = "openai" # openai-compatible API




def _build_model():
    print(f"Building model with {config['models'][config['modelIdx']]}, temperature={config['temperature']}, timeout={config['timeout']}s")
    return init_chat_model(
        model=config["models"][config["modelIdx"]],
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
    start_tag = "#### "
    start_idx = answer.find(start_tag)
    if start_idx != -1:
        rationale = answer[:start_idx].strip()
        answer = answer[start_idx + len(start_tag):].strip()
    else:
        rationale = ""
        answer = answer.strip()
    return answer, rationale


def _record_usage(response: str, include: bool = True) -> int:
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


def llm(text: str, cot: bool = False, few_shots: int = 0) -> tuple[int, str, int, float]:
    """Run a single-turn hypothesis improvement using only an LLM.
    Args:
        text: The input text to rate the CEFR level of.
        cot: Whether to include a chain-of-thought rationale in the prompt. Defaults to False.
        few_shots: The number of few-shot examples to include in the prompt. Defaults to 0.
    Returns:
        - The final CEFR level predicted by the model in integer labels.
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
        system_prompt = SYSTEM_PROMPT_COT if cot else SYSTEM_PROMPT_TEMPLATE

    model = _build_model()

    token_usage = 0

    conversation: List[HumanMessage | AIMessage | SystemMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the text to rate the CEFR level of:\n{text}"),
    ]

    assistant_message = model.invoke(conversation)
    token_usage += _record_usage(assistant_message)
    assistant_text = _message_text(assistant_message)

    answer, rationale = _extract_final_answer(assistant_text)
    try:
        answer = int(answer)
    except ValueError:
        raise ValueError(f"Final answer is not a valid integer: '{answer}'")
    return answer, rationale, token_usage, time.time() - now



if __name__ == "__main__":
    # Load the dataset
    train_df, test_df = load_data_split()
    text = test_df["text"][0]
    print(f"Input text:\n{text}\n")
    answer, rationale, token_usage, elapsed_time = llm(text, cot=True, few_shots=3)
    if answer is not None:
        print(f"Predicted CEFR level (integer label): {answer}")
        print("\nRationale:\n")
        print(rationale)
    print(f"\nApproximate tokens used: {token_usage}")
    print(f"Total time taken for the workflow run: {elapsed_time:.2f} seconds")