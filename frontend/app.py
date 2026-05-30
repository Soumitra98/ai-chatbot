import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")
st.caption("Powered by HuggingFace · LangChain · FastAPI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                res = requests.post(
                    API_URL, json={"message": prompt}, timeout=60)
                reply = res.json().get("reply", "Error: no response.")
            except Exception as e:
                reply = f"Connection error: {e}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})