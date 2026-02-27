from functools import lru_cache
import random

SYSTEM_PROMPT_TEMPLATE = "You are a professional linguist specializing in language proficiency assessment. " + \
"Your task is to evaluate the CEFR level of a given text passage. The CEFR levels are as follows:" + \
"""
A1 (Beginner): Can understand and use familiar everyday expressions and very basic phrases.
A2 (Elementary): Can understand sentences and frequently used expressions related to areas of most immediate relevance.
B1 (Intermediate): Can understand the main points of clear standard input on familiar matters regularly encountered in work, school, leisure, etc.
B2 (Upper Intermediate): Can understand the main ideas of complex text on both concrete and abstract topics, including technical discussions in their field of specialization.
C1 (Advanced): Can understand a wide range of demanding, longer texts, and recognize implicit meaning.
C2 (Proficient): Can understand with ease virtually everything heard or read. Can summarize information from different spoken and written sources, reconstructing arguments and accounts in a coherent presentation.

When evaluating the CEFR level, consider the following factors:
- Vocabulary: Assess the range and complexity of vocabulary used in the text.
- Grammar: Evaluate the variety and accuracy of grammatical structures.
- Cohesion and Coherence: Analyze how well the text is organized and how ideas are connected.
- Task Achievement: Consider how well the text fulfills its communicative purpose.
"""

INSTRUCTION = """
Based on the above criteria, read the provided text passage and determine its CEFR level. Provide a single CEFR level (A1, A2, B1, B2, C1, or C2) as your final answer following the delimiter "#### ". 
For example:
### A1
"""

INSTRUCTION_COT = """
Based on the above criteria, read the provided text passage and determine its CEFR level. Please provide a step-by-step rationale for your CEFR level assessment, breaking down your evaluation into several components.
Always provide your CEFR level assessment following a rationale explaining your assessment. Do not embed the final CEFR level within the rationale. Instead, clearly separate the rationale and the final answer using the delimiter "#### ".
Format your response as follows:
<your rationale>#### <your CEFR level>

For example:
The text demonstrates a limited range of vocabulary and simple sentence structures, which are characteristic of a beginner level. The ideas are not well connected, and there are frequent grammatical errors. #### A1
"""

SYSTEM_PROMPT_BASE = SYSTEM_PROMPT_TEMPLATE + INSTRUCTION

SYSTEM_PROMPT_COT = SYSTEM_PROMPT_BASE + INSTRUCTION_COT

@lru_cache(maxsize=1)
def _get_train_records() -> tuple[dict, ...]:
    """Load and cache training records for few-shot prompt construction.

    Cached in-memory because the dataset is small and the few-shot prompt builder is called
    for every sample when `few_shots > 0`.
    """
    import sys

    sys.path.append("..")
    from data_loader import load_data_split

    train_df, _ = load_data_split()
    # Tuple is immutable/hash-stable for caching; contains dicts (records).
    return tuple(train_df.to_dict(orient="records"))


def system_prompt_few_shots(n: int = 1, cot: bool = False) -> str:
    """
    Construct a few-shots system prompt by appending n randomly selected examples of CEFR level assessments to the base system prompt template. 
    Each example should include a corresponding CEFR level label formatted as specified in the template.
    Since no rationale is provided for selecting examples, we will randomly sample n examples from the dataset to include in the prompt.
    The examples are from the training dataset to avoid data leakage from the test set.
    Args:
        n: The number of examples to include in the few-shots prompt. Defaults to 1.
        cot: Whether to include a chain-of-thought rationale in the prompt. Defaults to False.
    Returns:
        A string containing the complete system prompt with n examples appended.
    """
    train_records = _get_train_records()
    examples = random.sample(list(train_records), k=min(n, len(train_records)))
    example_strs = []
    for example in examples:
        example_strs.append(f"\nText:\n{example['text']}\nResponse: <skipping rationale>#### {example['cefr_level']}")
    return SYSTEM_PROMPT_TEMPLATE + "\n\nHere are some examples of CEFR level assessments:\n" + \
        "\n".join(example_strs) + (INSTRUCTION_COT if cot else INSTRUCTION)



