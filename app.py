import streamlit as st
from query import answer

st.set_page_config(
    page_title="MindMap Sales Assistant",
    page_icon="💼",
    layout="centered",
)

st.title("MindMap Sales Assistant")
st.caption("Ask about case studies, capabilities, ROI metrics, and use cases.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(answer(prompt, st.session_state.history))

    # Save to display history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Update conversation history for RAG context (last 10 messages)
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": response})
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]
