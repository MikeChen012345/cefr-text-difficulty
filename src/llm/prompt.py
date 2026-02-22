SYSTEM_PROMPT_TEMPLATE = "You are a professional linguist specializing in language proficiency assessment. " + \
"Your task is to evaluate the CEFR level of a given text passage and provide a rationale for your assessment. The CEFR levels are as follows:" + \
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
Always provide your CEFR level assessment following a rationale explaining your assessment. Do not embed the final CEFR level within the rationale. Instead, clearly separate the rationale and the final answer using the delimiter "#### ".
Format your response as follows:
<your rationale>#### <your CEFR level>

For example:
The text demonstrates a limited range of vocabulary and simple sentence structures, which are characteristic of a beginner level. The ideas are not well connected, and there are frequent grammatical errors. #### A1
"""

COT = "\n\nPlease provide a step-by-step rationale for your CEFR level assessment, breaking down your evaluation into several components."



SYSTEM_PROMPT_BASE = SYSTEM_PROMPT_TEMPLATE + INSTRUCTION

SYSTEM_PROMPT_COT = SYSTEM_PROMPT_BASE + COT + INSTRUCTION

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
    import numpy as np
    import sys
    sys.path.append("..")
    from data_loader import load_data_split
    train_df, _ = load_data_split()
    examples = np.random.choice(train_df.to_dict(orient="records"), size=min(n, len(train_df)), replace=False)
    example_strs = []
    for example in examples:
        example_strs.append(f"\nText:\n{example['text']}\nResponse: <skipping rationale>#### {example['cefr_level']}")
    return SYSTEM_PROMPT_TEMPLATE + "\n\nHere are some examples of CEFR level assessments:\n" + \
        "\n".join(example_strs) + (COT if cot else "") + INSTRUCTION



