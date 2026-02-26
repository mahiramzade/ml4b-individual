from langchain_openai import ChatOpenAI
import streamlit as st
from openai import AuthenticationError
from typing import TypedDict
from langchain.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.callbacks import get_openai_callback


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

st.set_page_config(page_title="LLM Chat", page_icon="💬", layout="wide")

# Token limit configuration (in tokens)
TOKEN_LIMIT = 20000

# Initialize chat history and token tracking
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Let's start chatting! 👇"}
    ]
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "prompt_tokens" not in st.session_state:
    st.session_state.prompt_tokens = 0
if "completion_tokens" not in st.session_state:
    st.session_state.completion_tokens = 0
if "input_cost" not in st.session_state:
    st.session_state.input_cost = 0.0
if "output_cost" not in st.session_state:
    st.session_state.output_cost = 0.0

# Display token usage in sidebar
with st.sidebar:
    st.subheader("🔑 API Key")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="openai_api_key_input",
        help="Your OpenAI API key is required to use this service.",
    )
    st.session_state.openai_api_key = api_key.strip() if api_key else ""
    if not st.session_state.openai_api_key:
        st.error("OpenAI API key is required. Enter your key above to continue.")

    st.subheader("Model")
    selected_model = st.selectbox(
        "Select model",
        options=list(MODEL_CONFIG.keys()),
        key="model_select",
    )
    st.session_state.selected_model = selected_model

    st.subheader("Token Usage")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.caption("Total")
        st.caption(f"{st.session_state.total_tokens:,}")
    with tc2:
        st.caption("Prompt")
        st.caption(f"{st.session_state.prompt_tokens:,}")
    with tc3:
        st.caption("Completion")
        st.caption(f"{st.session_state.completion_tokens:,}")
    prog_col, btn_col = st.columns([1, 1])
    with prog_col:
        st.progress(
            min(st.session_state.total_tokens / TOKEN_LIMIT, 1.0),
            text=f"{st.session_state.total_tokens:,} / {TOKEN_LIMIT:,}",
        )
    with btn_col:
        if st.button("Reset Chat", type="primary"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Let's start chatting! 👇"}
            ]
            st.session_state.total_tokens = 0
            st.session_state.prompt_tokens = 0
            st.session_state.completion_tokens = 0
            st.session_state.input_cost = 0.0
            st.session_state.output_cost = 0.0
            st.rerun()

    st.subheader("💰 Cost")
    total_cost = st.session_state.input_cost + st.session_state.output_cost
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.caption("Input")
        st.caption(f"${st.session_state.input_cost:.4f}")
    with cc2:
        st.caption("Output")
        st.caption(f"${st.session_state.output_cost:.4f}")
    with cc3:
        st.caption("Total")
        st.caption(f"${total_cost:.4f}")

    if st.session_state.total_tokens >= TOKEN_LIMIT:
        st.error(f"⚠️ Token limit reached ({TOKEN_LIMIT:,})")
        st.caption("Reset the chat to continue.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Require API key before accepting input
if not st.session_state.get("openai_api_key"):
    st.info("👆 Enter your OpenAI API key in the sidebar to start chatting.")
    st.stop()

# Check token limit before accepting input
if st.session_state.total_tokens >= TOKEN_LIMIT:
    warning_msg = (
        f"⚠️ Token limit reached ({TOKEN_LIMIT:,} tokens). "
        "Please reset the chat to continue."
    )
    st.warning(warning_msg)
    st.stop()

# Accept user input
if prompt := st.chat_input("Ask me about any industry..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Convert session state messages to LangChain messages for agent
        agent_messages = [SYSTEM_MESSAGE]
        # Exclude the just-added user message
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                agent_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                agent_messages.append(AIMessage(content=msg["content"]))
        agent_messages.append(HumanMessage(content=prompt))

        # Create placeholder for thinking indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.status("🤔 Thinking...")

        # Create model and agent with user's API key and selected model
        agent = create_model_and_agent(
            api_key=st.session_state.openai_api_key,
            model_name=st.session_state.selected_model,
        )

        # Stream agent response with token tracking
        full_response = ""
        tokens_received = False

        try:
            # Use OpenAI callback to track token usage
            with get_openai_callback() as cb:
                for event in agent.stream(
                    {"messages": agent_messages},
                    stream_mode=["messages"],
                ):
                    did_yield = False

                    if isinstance(event, tuple) and len(event) == 2:
                        stream_mode, data = event
                        if stream_mode == "messages" and isinstance(data, tuple):
                            chunk, metadata = data

                            if isinstance(chunk, AIMessageChunk):
                                # Handle different content formats
                                chunk_text = ""
                                if isinstance(chunk.content, list):
                                    for content_item in chunk.content:
                                        if (
                                            isinstance(content_item, dict)
                                            and content_item.get("type") == "text"
                                        ):
                                            text = content_item.get("text", "")
                                            if text:
                                                chunk_text += text
                                                did_yield = True
                                        elif isinstance(content_item, str):
                                            if content_item:
                                                chunk_text += content_item
                                                did_yield = True
                                elif isinstance(chunk.content, str) and chunk.content:
                                    chunk_text = chunk.content
                                    did_yield = True

                                if did_yield:
                                    if not tokens_received:
                                        thinking_placeholder.empty()
                                        tokens_received = True
                                    full_response += chunk_text
                                    # Add a blinking cursor to simulate typing
                                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Update token usage from callback
            if cb.total_tokens > 0:
                st.session_state.prompt_tokens += cb.prompt_tokens
                st.session_state.completion_tokens += cb.completion_tokens
                st.session_state.total_tokens += cb.total_tokens

                # Calculate and update costs based on selected model
                model_config = MODEL_CONFIG[st.session_state.selected_model]
                input_cost = (
                    cb.prompt_tokens / 1_000_000
                ) * model_config["input_cost_per_million"]
                output_cost = (
                    cb.completion_tokens / 1_000_000
                ) * model_config["output_cost_per_million"]
                st.session_state.input_cost += input_cost
                st.session_state.output_cost += output_cost

        except AuthenticationError:
            thinking_placeholder.empty()
            full_response = "The API key you entered is incorrect. Please check your key in the sidebar and try again."
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Check if limit exceeded after this response
    if st.session_state.total_tokens >= TOKEN_LIMIT:
        st.error(
            f"⚠️ Token limit reached ({TOKEN_LIMIT:,} tokens). "
            "Please use the reset button in the sidebar to continue."
        )

    # Rerun to update sidebar stats in real-time
    st.rerun()