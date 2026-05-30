# app.py  ←  single file for Streamlit Community Cloud deployment

import streamlit as st
from huggingface_hub import InferenceClient
from langchain_core.chat_history import InMemoryChatMessageHistory
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 AI Chatbot")
st.caption("Powered by Qwen2.5-7B · HuggingFace · LangChain")

# ── Load credentials from Streamlit secrets ───────────────────────────────────
HF_TOKEN = st.secrets["HF_TOKEN"]
MODEL_ID = st.secrets.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

# ── Session state: memory + message display ───────────────────────────────────
if "lc_history" not in st.session_state:
    st.session_state.lc_history = InMemoryChatMessageHistory()

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# ── InferenceClient (cached so it isn't rebuilt on every rerun) ───────────────


@st.cache_resource
def get_client(token: str, model: str) -> InferenceClient:
    return InferenceClient(model=model, token=token)


client = get_client(HF_TOKEN, MODEL_ID)

# ── Helper: call HuggingFace and update LangChain memory ─────────────────────


def get_reply(user_message: str) -> str:
    history = st.session_state.lc_history

    # Build messages list
    messages = [
        {"role": "system", "content": "You are a helpful and friendly AI assistant."}]
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    response = client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )
    reply = response.choices[0].message.content

    # Persist to LangChain memory
    history.add_user_message(user_message)
    history.add_ai_message(reply)
    return reply


# ── Render existing chat history ──────────────────────────────────────────────
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type your message…"):
    # Show user bubble immediately
    st.session_state.display_messages.append(
        {"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and show assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = get_reply(prompt)
            except Exception as e:
                reply = f"⚠️ Error: {e}"
        st.markdown(reply)

    st.session_state.display_messages.append(
        {"role": "assistant", "content": reply})
