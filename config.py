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


# ── Session state init ────────────────────────────────────────────────────────

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if "active_chat" not in st.session_state:
    st.session_state.active_chat = 0


def current_chat():
    return st.session_state.chats[st.session_state.active_chat]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Chats")
    if st.button("+ New Chat", use_container_width=True):
        st.session_state.chats.append({"title": "New Chat", "messages": [], "history": []})
        st.session_state.active_chat = len(st.session_state.chats) - 1
        save_chats()
        st.rerun()

    st.divider()

    for i, chat in enumerate(st.session_state.chats):
        label = chat["title"]
        if i == st.session_state.active_chat:
            st.markdown(f"**> {label}**")
        else:
            if st.button(label, key=f"chat_{i}", use_container_width=True):
                st.session_state.active_chat = i
                st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("MindMap Sales Assistant")
st.caption("Ask about case studies, capabilities, ROI metrics, and use cases.")


def render_sources(sources: list, key_prefix: str = "") -> None:
    if not sources:
        return
    label = f"Sources ({len(sources)} file{'s' if len(sources) != 1 else ''})"
    with st.expander(label, expanded=False):
        for src in sources:
            fname = src["file_name"]
            fpath = src.get("file_path", "")
            doc_type = src.get("doc_type", "").replace("_", " ").title()
            verticals = src.get("industry_vertical", [])
            meta = doc_type
            if verticals:
                meta += "  ·  " + ", ".join(verticals)

            full_path = str(DATA_DIR / fpath) if fpath and not os.path.isabs(fpath) else fpath

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{fname}**  \n*{meta}*")
            with col2:
                if full_path and os.path.exists(full_path):
                    ext = os.path.splitext(fname)[1].lower()
                    mime_map = {
                        ".pdf":  "application/pdf",
                        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    }
                    mime = mime_map.get(ext, "application/octet-stream")
                    with open(full_path, "rb") as f:
                        file_bytes = f.read()
                    st.download_button(
                        label="Download",
                        data=file_bytes,
                        file_name=fname,
                        mime=mime,
                        key=f"{key_prefix}_{fname}",
                    )


# Render existing messages in active chat
chat = current_chat()
for i, msg in enumerate(chat["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"], key_prefix=f"c{st.session_state.active_chat}_msg{i}")


# Chat input
if prompt := st.chat_input("Ask something..."):
    chat = current_chat()

    # Set chat title from first user message
    if not chat["messages"]:
        chat["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    is_greeting = prompt.strip().lower().rstrip("!.,") in _GREETINGS
    chunks = [] if is_greeting else retrieve(prompt)
    sources = get_sources(chunks)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_answer(prompt, chunks, chat["history"]))
        render_sources(sources, key_prefix=f"c{st.session_state.active_chat}_current")

    chat["messages"].append({"role": "assistant", "content": response, "sources": sources})

    chat["history"].append({"role": "user", "content": prompt})
    chat["history"].append({"role": "assistant", "content": response})
    if len(chat["history"]) > 10:
        chat["history"] = chat["history"][-10:]

    save_chats()
