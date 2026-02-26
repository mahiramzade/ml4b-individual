"""
LLM-as-judge evaluator: chain-based batch evaluation (yes/no per query–response pair).

  • create_eval_chain(llm, model_name_or_id): Gemma → single user message; else system + user.
  • evaluate_batch(llm, inputs, assistant_system_prompt, ...): returns list[bool].
  • evaluate_dataframe(df, llm, ...): returns df with "adequate" column (raw yes/no).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

if TYPE_CHECKING:
    import pandas as pd

# System prompt for the generative AI evaluator.
EVALUATOR_SYSTEM_PROMPT = """You are an impartial evaluator. Your task is to judge whether an assistant's response adequately addresses the user's query given the assistant's own instructions (system prompt).

Say yes if: (a) industry/market query and the response has a Summary plus Sources with links—even if brief; (b) off-topic query and the assistant clearly refuses or redirects. Be lenient: minor format differences are fine.

Say no only if: the response clearly ignores the query, gives wrong-domain content, or violates the assistant's role.

You must reply with exactly one word: yes or no. No explanation, no reasoning, no other text."""

# User prompt template; curly brackets are filled with each query–response pair.
# assistant_system_prompt: the instructions the assistant was given; use so the evaluator judges adequacy relative to them.
EVALUATOR_USER_PROMPT_TEMPLATE = """Assistant's instructions (system prompt):
{assistant_system_prompt}

User query:
{user_query}

Assistant response:
{assistant_response}

Was the user query adequately addressed by the assistant given its instructions? Reply with exactly one word: yes or no."""


def create_eval_chain(
    llm: BaseChatModel,
    model_name_or_id: str = "",
) -> RunnableSequence:
    """
    Build evaluator chain: prompt_template | llm.
    If "gemma" is in model_name_or_id (case-insensitive), system and user prompts
    are merged into a single user message; otherwise use separate system and user messages.
    """
    merged_content = f"{EVALUATOR_SYSTEM_PROMPT}\n\n{EVALUATOR_USER_PROMPT_TEMPLATE}"
    if "gemma" in (model_name_or_id or "").lower():
        prompt_template = ChatPromptTemplate.from_messages([
            ("user", merged_content),
        ])
    else:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", EVALUATOR_SYSTEM_PROMPT),
            ("user", EVALUATOR_USER_PROMPT_TEMPLATE),
        ])
    return prompt_template | llm


def evaluate_batch(
    llm: BaseChatModel,
    inputs: list[dict],
    assistant_system_prompt: str = "",
    model_name_or_id: str = "",
) -> list[bool]:
    """
    Run evaluator on multiple query–response pairs. Each input dict must have
    "query" and "response" (or "user_query" and "assistant_response").
    assistant_system_prompt: passed in by the caller (e.g. main.SYSTEM_MESSAGE.content).
    Returns list of bools (True = adequate). Uses create_eval_chain and chain.batch().
    """
    chain = create_eval_chain(llm, model_name_or_id=model_name_or_id)
    normalized = []
    for row in inputs:
        q = row.get("user_query") or row.get("query", "")
        r = row.get("assistant_response") or row.get("response", "")
        normalized.append({
            "assistant_system_prompt": assistant_system_prompt or "(Not provided)",
            "user_query": q,
            "assistant_response": r,
        })
    results = chain.batch(normalized)
    def _parse_adequate(msg) -> bool:
        text = (getattr(msg, "content", None) or str(msg)).strip().lower()
        if not text:
            return False
        # Strip markdown/code fences; some models wrap output
        clean = text.strip("`\n\t ")
        words = [w.rstrip(".,;:") for w in clean.split() if w]
        if not words:
            return False
        first, last = words[0], words[-1]
        # Many models output "yes" or "no" first; some add reasoning and put verdict last
        if first == "yes" or last == "yes" or clean.startswith("yes"):
            return True
        if first == "no" or last == "no" or clean.startswith("no"):
            return False
        return False
    return [_parse_adequate(r) for r in results]


def evaluate_dataframe(
    df: pd.DataFrame,
    llm: BaseChatModel,
    assistant_system_prompt: str = "",
    model_name_or_id: str = "",
    query_col: str = "query",
    response_col: str = "response",
) -> pd.DataFrame:
    """
    Run evaluator on each row of a DataFrame. Expects columns query_col and response_col
    (default "query", "response"). assistant_system_prompt: from caller (e.g. main.SYSTEM_MESSAGE.content).
    Returns a copy of the DataFrame with an "adequate" column (raw "yes"/"no" per row).
    """
    import pandas as pd
    inputs = [{query_col: row[query_col], response_col: row[response_col]} for _, row in df.iterrows()]
    # Map to keys evaluate_batch expects
    normalized = [
        {
            "assistant_system_prompt": assistant_system_prompt or "(Not provided)",
            "user_query": inp[query_col],
            "assistant_response": inp[response_col],
        }
        for inp in inputs
    ]
    chain = create_eval_chain(llm, model_name_or_id=model_name_or_id)
    results = chain.batch(normalized)
    out = df.copy()
    out["adequate"] = [getattr(r, "content", str(r)) for r in results]
    return out
