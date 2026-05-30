from huggingface_hub import InferenceClient
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

# Session memory store
_store: dict = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


# Single shared InferenceClient pointed at Qwen2.5-7B-Instruct
client = InferenceClient(
    model=os.getenv("MODEL_ID"),   # Qwen/Qwen2.5-7B-Instruct
    token=os.getenv("HF_TOKEN"),
)


def chat(message: str, session_id: str = "default") -> str:
    history = get_session_history(session_id)

    # Build full message list from LangChain history
    messages = [
        {"role": "system", "content": "You are a helpful and friendly AI assistant."}]

    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})

    messages.append({"role": "user", "content": message})

    # Call HuggingFace Inference API directly
    response = client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )

    reply = response.choices[0].message.content

    # Persist to LangChain memory
    history.add_user_message(message)
    history.add_ai_message(reply)

    return reply
