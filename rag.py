from __future__ import annotations

import os
from typing import Generator, List, Dict, Optional

import cohere
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
)

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

RERANK_MODEL = "rerank-english-v3.0"
RERANK_TOP_N = 5
RETRIEVAL_TOP_K = 20

_openai: OpenAI | None = None
_qdrant: QdrantClient | None = None
_cohere: cohere.Client | None = None


def openai_client() -> OpenAI:
    global _openai

    if _openai is None:
        _openai = OpenAI(api_key=OPENAI_API_KEY)

    return _openai


def qdrant_client() -> QdrantClient:
    global _qdrant

    if _qdrant is None:
        kwargs = {"url": QDRANT_URL}

        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY

        _qdrant = QdrantClient(**kwargs)

    return _qdrant


def cohere_client() -> cohere.Client:
    global _cohere

    if _cohere is None:
        _cohere = cohere.Client(api_key=COHERE_API_KEY)

    return _cohere


def _embed_query(text: str) -> List[float]:
    resp = openai_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.replace("\n", " "),
    )

    return resp.data[0].embedding


def retrieve(user_query: str) -> List[Dict]:
    """Qdrant search → Cohere rerank."""

    query_vec = _embed_query(user_query)

    results = qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=RETRIEVAL_TOP_K,
        with_payload=True,
    ).points

    if not results:
        return []

    chunks = [r.payload for r in results]

    rerank_resp = cohere_client().rerank(
        model=RERANK_MODEL,
        query=user_query,
        documents=[c.get("text", "") for c in chunks],
        top_n=RERANK_TOP_N,
    )

    reranked = []

    for hit in rerank_resp.results:
        chunk = chunks[hit.index]
        chunk["_rerank_score"] = round(hit.relevance_score, 4)
        reranked.append(chunk)

    return reranked


ANSWER_SYSTEM = """
You are a sales assistant for MindMap Digital — an RPA and AI automation consultancy.

You help the internal sales team find relevant case studies, capabilities, ROI metrics, 
and use cases from MindMap's sales collateral.

CRITICAL RULE — NO HALLUCINATION:
  - You MUST only use information that appears word-for-word in the provided context chunks.                                                   
  - If a client name, metric, percentage, or outcome is not explicitly stated in the context, DO NOT include it.                               
  - Do NOT invent, infer, or generalise any facts, numbers, client names, or outcomes.                                                           - If the context does not contain enough information to answer, say exactly: "No specific data available in current collateral. Check with   
  the delivery team."                                                                                                                             
  RESPONSE LENGTH:                                                                                                                             
  - DEFAULT: Short — maximum 3 to 4 bullet points or 2 to 3 sentences.
  - IN-DEPTH ONLY when the user explicitly says "explain in detail", "deep dive", "elaborate", "tell me more", or similar.                     
                                                                                                                                                 Other rules:                                                                                                                                 
  - Use bullet points or short sentences — no markdown headers (no #, ##, ###)                                                                 
  - Do not use emojis or informal language                                                                                                     
  - Do not include source citations or document references in your answer                                                                      
  - When listing ROI metrics, quote them exactly as they appear in the context 
"""


def _format_context(chunks: List[Dict]) -> str:

    parts = []

    for i, chunk in enumerate(chunks, 1):

        parts.append(
            f"[Chunk {i}]\n"
            f"File: {chunk.get('file_name', 'Unknown')}\n"
            f"Type: {chunk.get('doc_type', 'Unknown')}\n"
            f"Section: {chunk.get('section_type', 'Unknown')}\n"
            f"Verticals: {', '.join(chunk.get('industry_vertical', []))}\n\n"
            f"{chunk.get('text', '')}"
        )

    return "\n\n---\n\n".join(parts)


def get_sources(chunks: List[Dict]) -> List[Dict]:
    """Extract unique source files from retrieved chunks."""

    seen: dict = {}

    for chunk in chunks:

        fname = chunk.get("file_name", "")

        if not fname:
            continue

        if fname not in seen:
            seen[fname] = {
                "file_name": fname,
                "file_path": chunk.get("file_path", ""),
                "file_url": chunk.get("file_url", ""),
                "doc_type": chunk.get("doc_type", ""),
                "industry_vertical": chunk.get("industry_vertical", []),
                "source_folder": chunk.get("source_folder", ""),
                "page_numbers": set(),
            }

        seen[fname]["page_numbers"].update(chunk.get("page_numbers", []))

    sources = []

    for src in seen.values():
        src["page_numbers"] = sorted(src["page_numbers"])
        sources.append(src)

    return sources


_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "howdy",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
}


def stream_answer(
    user_query: str,
    chunks: List[Dict],
    conversation_history: List[Dict],
) -> Generator[str, None, None]:

    if user_query.strip().lower().rstrip("!.,") in _GREETINGS:

        yield (
            "Hello! How can I help you with MindMap Digital's sales collateral? "
            "You can ask about case studies, capabilities, ROI metrics, "
            "or specific industry use cases."
        )
        return

    if not chunks:
        yield "No specific data available in current collateral. Check with the delivery team."
        return

    context = _format_context(chunks)

    messages = [{"role": "system", "content": ANSWER_SYSTEM}]
    messages.extend(conversation_history[-10:])

    messages.append(
        {
            "role": "user",
            "content": f"Context from sales collateral:\n\n{context}\n\nQuestion: {user_query}",
        }
    )

    stream = openai_client().chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content

        if delta:
            yield delta


def answer(
    user_query: str,
    conversation_history: List[Dict],
) -> Generator[str, None, None]:

    if user_query.strip().lower().rstrip("!.,") in _GREETINGS:

        yield (
            "Hello! How can I help you with MindMap Digital's sales collateral? "
            "You can ask about case studies, capabilities, ROI metrics, "
            "or specific industry use cases."
        )
        return

    chunks = retrieve(user_query)

    yield from stream_answer(user_query, chunks, conversation_history)


# ─────────────────────────────────────────────────────
# CLI Chat
# ─────────────────────────────────────────────────────

def chat_cli():

    print("\n" + "=" * 60)
    print("  MindMap Sales Collateral Assistant")
    print("  Type 'exit' to quit")
    print("=" * 60 + "\n")

    history: List[Dict] = []

    while True:

        try:
            user_input = input("You: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)

        full_response = ""

        for token in answer(user_input, history):
            print(token, end="", flush=True)
            full_response += token

        print("\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})

        if len(history) > 10:
            history = history[-10:]


if __name__ == "__main__":
    chat_cli()
