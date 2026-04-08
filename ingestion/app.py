import os
import json
import secrets as _secrets_mod
from datetime import datetime
import msal
import streamlit as st
from rag import retrieve, get_sources, stream_answer, generate_title, generate_summary, _GREETINGS
from config import DATA_DIR

st.set_page_config(
    page_title="MindMap Sales Assistant",
    page_icon="💼",
    layout="centered",
)

# ── Microsoft OAuth ────────────────────────────────────────────────────────────

_MS = st.secrets["microsoft"]
_MSAL_APP = msal.ConfidentialClientApplication(
    _MS["client_id"],
    authority=f"https://login.microsoftonline.com/{_MS['tenant_id']}",
    client_credential=_MS["client_secret"],
)
_SCOPES = ["User.Read"]
_ALLOWED_DOMAIN = "mindmapdigital.com"


def _auth_url() -> str:
    if "oauth_state" not in st.session_state:
        st.session_state.oauth_state = _secrets_mod.token_urlsafe(16)
    return _MSAL_APP.get_authorization_request_url(
        scopes=_SCOPES,
        redirect_uri=_MS["redirect_uri"],
        state=st.session_state.oauth_state,
    )


def _handle_callback():
    """Exchange the auth code for a token and store user info in session."""
    params = st.query_params
    code = params.get("code")
    if not code:
        return

    result = _MSAL_APP.acquire_token_by_authorization_code(
        code, scopes=_SCOPES, redirect_uri=_MS["redirect_uri"]
    )
    st.query_params.clear()

    if "error" in result:
        st.error(f"Sign-in failed: {result.get('error_description', result['error'])}")
        st.rerun()

    claims = result.get("id_token_claims", {})
    email = claims.get("preferred_username") or claims.get("email", "")
    name = claims.get("name") or email.split("@")[0]

    st.session_state.ms_user = {"email": email, "name": name}
    st.session_state.username = email.split("@")[0].lower()
    st.session_state.name = name
    st.rerun()


# Process callback if Microsoft redirected back with a code
_handle_callback()

# Gate: show login screen if not authenticated
if not st.session_state.get("ms_user"):
    st.markdown("## MindMap Sales Assistant")
    st.markdown("Sign in with your Microsoft account to continue.")
    _url = _auth_url()
    st.markdown(
        f'<a href="{_url}" target="_top" style="display:inline-block;width:100%;text-align:center;'
        f'padding:0.5em 1em;background:#0078d4;color:white;border-radius:6px;'
        f'text-decoration:none;font-size:1em;font-weight:600;box-sizing:border-box;">'
        f'Sign in with Microsoft</a>',
        unsafe_allow_html=True,
    )
    st.stop()

def _chats_file() -> str:
    username = st.session_state.get("username", "default")
    return str(DATA_DIR.parent / f"chats_{username}.json")


def load_chats():
    chats_file = _chats_file()
    if os.path.exists(chats_file):
        try:
            with open(chats_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return [{"title": "New Chat", "messages": [], "history": []}]


def save_chats():
    with open(_chats_file(), "w") as f:
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
    user_info = st.session_state.get("ms_user", {})
    st.caption(f"Signed in as **{user_info.get('name', '')}**")
    if st.button("Sign out", use_container_width=True):
        for key in ("ms_user", "username", "name", "oauth_state"):
            st.session_state.pop(key, None)
        st.rerun()
    st.title("Chats")
    if st.button("+ New Chat", use_container_width=True):
        st.session_state.chats.append({"title": "New Chat", "messages": [], "history": []})
        st.session_state.active_chat = len(st.session_state.chats) - 1
        save_chats()
        st.rerun()

    st.divider()

    if st.button("End Chat & Summary", use_container_width=True):
        chat = current_chat()
        if chat.get("messages"):
            with st.spinner("Generating summary..."):
                summary = generate_summary(chat["messages"])
            transcript_lines = []
            for msg in chat["messages"]:
                role = "You" if msg["role"] == "user" else "Assistant"
                transcript_lines.append(f"{role}:\n{msg['content']}\n")
            title = chat.get("title", "Chat")
            txt_content = (
                f"{title}\n{'=' * len(title)}\n\nSummary\n-------\n{summary}\n\n"
                f"Conversation\n------------\n" + "\n".join(transcript_lines)
            )
            date_str = datetime.now().strftime("%d %b %Y")
            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip()
            st.session_state.end_chat_download = (f"{safe_title} - {date_str}.txt", txt_content)
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
            file_url = src.get("file_url", "")
            doc_type = src.get("doc_type", "").replace("_", " ").title()
            verticals = src.get("industry_vertical", [])
            meta = doc_type
            if verticals:
                meta += "  ·  " + ", ".join(verticals)

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{fname}**  \n*{meta}*")
            with col2:
                # Local run: serve file bytes directly
                full_path = str(DATA_DIR / fpath) if fpath and not os.path.isabs(fpath) else fpath
                if full_path and os.path.exists(full_path):
                    ext = os.path.splitext(fname)[1].lower()
                    mime_map = {
                        ".pdf":  "application/pdf",
                        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    }
                    with open(full_path, "rb") as f:
                        file_bytes = f.read()
                    st.download_button(
                        label="Download",
                        data=file_bytes,
                        file_name=fname,
                        mime=mime_map.get(ext, "application/octet-stream"),
                        key=f"{key_prefix}_{fname}",
                    )
                # Streamlit Cloud: link directly to Google Drive (browser handles download)
                elif file_url:
                    st.link_button("Download", file_url)


# Render existing messages in active chat
chat = current_chat()
for i, msg in enumerate(chat["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"], key_prefix=f"c{st.session_state.active_chat}_msg{i}")


# ── Input row ─────────────────────────────────────────────────────────────────

if "end_chat_download" not in st.session_state:
    st.session_state.end_chat_download = None

if st.session_state.end_chat_download:
    fname, content = st.session_state.end_chat_download
    st.download_button(
        label="Download chat as .txt",
        data=content.encode("utf-8"),
        file_name=fname,
        mime="text/plain",
        on_click=lambda: st.session_state.update(end_chat_download=None),
    )

prompt = st.chat_input("Ask something...")

# ── Send handler ──────────────────────────────────────────────────────────────

if prompt and prompt.strip():
    chat = current_chat()

    is_first_message = not chat["messages"]

    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    is_greeting = prompt.strip().lower().rstrip("!.,") in _GREETINGS

    _PRONOUNS = {"them", "they", "it", "this", "that", "their", "these", "those"}
    resolved_prompt = prompt
    words = prompt.lower().split()
    if not is_greeting and len(words) <= 8 and any(w in _PRONOUNS for w in words):
        history = chat.get("history", [])
        last_assistant = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
        )
        if last_assistant:
            resolved_prompt = f"{prompt} (context from previous answer: {last_assistant[:200]})"

    chunks = [] if is_greeting else retrieve(resolved_prompt)
    sources = get_sources(chunks)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_answer(prompt, chunks, chat["history"]))
        render_sources(sources, key_prefix=f"c{st.session_state.active_chat}_current")

    chat["messages"].append({"role": "assistant", "content": response, "sources": sources})

    if is_first_message and not is_greeting:
        try:
            chat["title"] = generate_title(prompt, response)
        except Exception:
            chat["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")
    elif is_first_message:
        chat["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

    chat["history"].append({"role": "user", "content": prompt})
    chat["history"].append({"role": "assistant", "content": response})
    if len(chat["history"]) > 10:
        chat["history"] = chat["history"][-10:]

    save_chats()
