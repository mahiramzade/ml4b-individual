"""
Entry point: run with `streamlit run main.py`

Contains main logic: agents, tools, model config, system message.
Streamlit UI lives in streamlit_ui.py.
"""
from typing import TypedDict

from langchain.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.retrievers import WikipediaRetriever


wikipedia_retriever = WikipediaRetriever(top_k_results=5)


class WikipediaPost(TypedDict):
    title: str
    url: str
    summary: str


@tool
def get_wikipedia_posts(industry_name: str) -> list[WikipediaPost]:
    """
    This tool returns most relevant posts from wikipedia for the given industry name.
    Args:
        industry_name: The name of the industry
    Returns:
        A list of Wikipedia posts for the given industry name.
    """
    result = wikipedia_retriever.invoke(industry_name)
    return [post.metadata for post in result]


MODEL_CONFIG = {
    "GPT 5.1": {
        "model_id": "gpt-5.1",
        "extra_kwargs": {"reasoning_effort": "none"},
        "input_cost_per_million": 1.25,
        "output_cost_per_million": 10.00,
    },
    "GPT 4.1 Mini": {
        "model_id": "gpt-4.1-mini-2025-04-14",
        "extra_kwargs": {},
        "input_cost_per_million": 0.40,
        "output_cost_per_million": 1.60,
    },
}


def create_model_and_agent(api_key: str, model_name: str):
    config = MODEL_CONFIG[model_name]
    model = ChatOpenAI(
        model=config["model_id"],
        temperature=0.0,
        api_key=api_key,
        **config["extra_kwargs"],
    )
    return create_agent(model, [get_wikipedia_posts])


SYSTEM_MESSAGE = SystemMessage(
    content="""\
You are a research assistant for a business analyst who conducts market research at a large corporation.
Your role is to support industry analysis, competitive intelligence, and strategic market assessments.

Do not guess the industry name—if unclear or ambiguous, ask the user to clarify.
Industry examples: pharmaceutical, biotechnology, medical devices, fashion, fintech, etc.

Use the given tools to gather information and produce a concise market research summary (max 500 words)
that is suitable for executive briefings and strategic planning. Base your analysis solely on the
sources provided—do not use external information or knowledge.
Only respond to industry and market research questions; decline irrelevant requests.

When an industry is provided, structure your response as follows:

Summary
(Analyze market dynamics, key players, trends, and factors relevant to corporate strategy.)

Sources:
* [Post title 1](url1)
* [Post title 2](url2)
* [Post title 3](url3)
* [Post title 4](url4)
* [Post title 5](url5)
"""
)

TOKEN_LIMIT = 20000

# Launch Streamlit UI (at end so streamlit_ui can import from this module)
import streamlit_ui  # noqa: F401, E402
