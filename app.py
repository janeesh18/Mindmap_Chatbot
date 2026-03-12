import os
import json
import streamlit as st
from rag import retrieve, get_sources, stream_answer, _GREETINGS
from config import DATA_DIR

st.set_page_config(
    page_title="MindMap Sales Assistant",
    page_icon="💼",
    layout="centered",
)

CHATS_FILE = str(DATA_DIR.parent / "chats.json")


def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return [{"title": "New Chat", "messages": [], "history": []}]


def save_chats():
    with open(CHATS_FILE, "w") as f:
        json.dump(st.session_state.chats, f)
