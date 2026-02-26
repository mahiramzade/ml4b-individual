"""Streamlit UI: sidebar, chat display, and agent streaming. Run via main.py."""

import streamlit as st
from openai import AuthenticationError
from langchain.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain_community.callbacks import get_openai_callback

# Core logic lives in main: model config, token limit, system prompt, and agent factory.
from main import (
    MODEL_CONFIG,
    TOKEN_LIMIT,
    SYSTEM_MESSAGE,
    create_model_and_agent,
)

# Page title, favicon, and use full width so chat has more space.
st.set_page_config(page_title="LLM Chat", page_icon="💬", layout="wide")

# --- Session state: chat history and token/cost tracking (persist across reruns). ---
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

# --- Sidebar: API key, model choice, token usage, cost, and reset. ---
with st.sidebar:
    st.subheader("🔑 API Key")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="openai_api_key_input",
        help="Your OpenAI API key is required to use this service.",
    )
    # Store trimmed key so the rest of the app can use it; show error if missing.
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

    # Token usage: total, prompt, completion in one row; progress bar and reset on the next row.
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
            # Clear chat and all token/cost counters, then rerun to refresh UI.
            st.session_state.messages = [
                {"role": "assistant", "content": "Let's start chatting! 👇"}
            ]
            st.session_state.total_tokens = 0
            st.session_state.prompt_tokens = 0
            st.session_state.completion_tokens = 0
            st.session_state.input_cost = 0.0
            st.session_state.output_cost = 0.0
            st.rerun()

    # Cost tracking: input, output, and total in one row (from MODEL_CONFIG pricing).
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

# --- Main area: show chat history (each message in its own bubble). ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Block new input until the user has set an API key.
if not st.session_state.get("openai_api_key"):
    st.info("👆 Enter your OpenAI API key in the sidebar to start chatting.")
    st.stop()

# Block new input when token limit is reached; user must reset in sidebar.
if st.session_state.total_tokens >= TOKEN_LIMIT:
    warning_msg = (
        f"⚠️ Token limit reached ({TOKEN_LIMIT:,} tokens). "
        "Please reset the chat to continue."
    )
    st.warning(warning_msg)
    st.stop()

# --- On new user message: append to history, show user bubble, then stream assistant reply. ---
if prompt := st.chat_input("Ask me about any industry..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Placeholder that we will update with streamed text (and a typing cursor).
        message_placeholder = st.empty()

        # Build LangChain message list: system prompt + prior chat + current user prompt.
        agent_messages = [SYSTEM_MESSAGE]
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                agent_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                agent_messages.append(AIMessage(content=msg["content"]))
        agent_messages.append(HumanMessage(content=prompt))

        # Show "Thinking..." until the first content chunk arrives.
        thinking_placeholder = st.empty()
        thinking_placeholder.status("🤔 Thinking...")

        # Create the agent (model + Wikipedia tool) using sidebar API key and model.
        agent = create_model_and_agent(
            api_key=st.session_state.openai_api_key,
            model_name=st.session_state.selected_model,
        )

        full_response = ""
        tokens_received = False

        try:
            # get_openai_callback() counts prompt/completion tokens for this run so we can bill and display.
            with get_openai_callback() as cb:
                # Stream agent events; we only care about "messages" events with assistant chunks.
                for event in agent.stream(
                    {"messages": agent_messages},
                    stream_mode=["messages"],
                ):
                    did_yield = False

                    if isinstance(event, tuple) and len(event) == 2:
                        stream_mode, data = event
                        if stream_mode == "messages" and isinstance(data, tuple):
                            chunk, metadata = data

                            # Extract plain text from AIMessageChunk (handles list or string content).
                            if isinstance(chunk, AIMessageChunk):
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

                                # Update UI: hide "Thinking..." on first token, append text and show cursor.
                                if did_yield:
                                    if not tokens_received:
                                        thinking_placeholder.empty()
                                        tokens_received = True
                                    full_response += chunk_text
                                    message_placeholder.markdown(full_response + "▌")

            # Final render without the typing cursor.
            message_placeholder.markdown(full_response)

            # Persist token counts and compute cost from this run using MODEL_CONFIG pricing.
            if cb.total_tokens > 0:
                st.session_state.prompt_tokens += cb.prompt_tokens
                st.session_state.completion_tokens += cb.completion_tokens
                st.session_state.total_tokens += cb.total_tokens

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

    # Save assistant reply to history so it appears on next rerun.
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    if st.session_state.total_tokens >= TOKEN_LIMIT:
        st.error(
            f"⚠️ Token limit reached ({TOKEN_LIMIT:,} tokens). "
            "Please use the reset button in the sidebar to continue."
        )

    # Rerun so sidebar token/cost and chat history update immediately.
    st.rerun()
