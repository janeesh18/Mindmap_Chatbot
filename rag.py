from __future__ import annotations

import os
import re
from typing import Dict, Generator, List, Optional

import cohere
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from config import (
    CLIENT_NAME_MAP,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
)

load_dotenv()

COHERE_API_KEY     = os.getenv("COHERE_API_KEY")
RERANK_MODEL       = "rerank-english-v3.0"
RERANK_TOP_N       = 5
RETRIEVAL_TOP_K    = 50
FALLBACK_THRESHOLD = 2  # Min results before fallback triggers

_openai: OpenAI | None        = None
_qdrant: QdrantClient | None  = None
_cohere: cohere.Client | None = None


# ── Lazy clients ──────────────────────────────────────────────────────────────

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


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_query(text: str) -> List[float]:
    resp = openai_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.replace("\n", " "),
    )
    return resp.data[0].embedding


# ── Intent detection ──────────────────────────────────────────────────────────

_CLIENT_TRIGGERS = {
    "client", "customer", "worked with", "worked for", "case study", "case studies",
    "which companies", "who have we", "our clients", "our customers", "past work",
    "intas", "piramal", "umc", "parker", "fellowship", "ingleside", "archcare",
    "wio", "uae bank", "kotak", "authbridge", "nga", "zurich",
}

_ROI_TRIGGERS = {
    "roi", "savings", "cost saving", "cost reduction", "fte reduction", "fte",
    "return on investment", "payback", "annual saving", "how much", "how many",
    "benefit", "result", "outcome", "impact", "achieved", "efficiency",
}

_VERTICAL_TRIGGERS: Dict[str, List[str]] = {
    "Healthcare":    ["healthcare", "hospital", "patient", "clinical", "medical"],
    "BFSI":          ["bank", "banking", "insurance", "fintech", "mortgage", "financial services"],
    "Pharma":        ["pharma", "pharmaceutical", "drug manufacturing"],
    "Telecom":       ["telecom", "telecommunications"],
    "Manufacturing": ["manufacturing", "production", "plant", "factory"],
    "SCM":           ["supply chain", "demand planning", "procurement", "inventory"],
    "HR":            ["human resources", "payroll", "recruitment", "onboarding"],
    "FA":            ["accounts payable", "accounts receivable", "invoice", "fp&a", "financial planning"],
    "Government":    ["government", "public sector"],
    "Logistics":     ["logistics", "transportation", "freight"],
    "Aviation":      ["aviation", "airline", "airport"],
    "Education":     ["education", "university", "higher education"],
}


def _detect_intent(query: str) -> Dict:
    q = query.lower()
    intent: Dict = {"client_query": False, "roi_query": False, "vertical": None}

    if any(kw in q for kw in _CLIENT_TRIGGERS):
        intent["client_query"] = True

    if any(kw in q for kw in _ROI_TRIGGERS):
        intent["roi_query"] = True

    for vertical, keywords in _VERTICAL_TRIGGERS.items():
        if any(kw in q for kw in keywords):
            intent["vertical"] = vertical
            break

    return intent


def _detect_specific_client(query: str) -> str | None:
    """Return canonical client name if a specific known client is mentioned in the query."""
    q = query.lower()
    for key, name in CLIENT_NAME_MAP.items():
        if key in q:
            return name
    return None


def _clean_query(query: str) -> str:
    """Remove brand name from query so embedding focuses on topic, not company name.
    Only strips if enough meaningful words remain after removal."""
    cleaned = re.sub(r"\bmindmap('?s?)?\b", "", query, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,?.")
    stopwords = {
        "what", "does", "do", "is", "are", "the", "a", "an",
        "tell", "me", "about", "how", "why", "who", "where", "when", "your",
    }
    meaningful = [w for w in cleaned.split() if w.lower() not in stopwords]
    return cleaned if len(meaningful) >= 2 else query


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(user_query: str) -> List[Dict]:
    """Qdrant search (with intent-based filters) → Cohere rerank."""
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchAny,
        MatchValue,
        PayloadSelectorInclude,
    )

    query_vec       = _embed_query(_clean_query(user_query))
    intent          = _detect_intent(user_query)
    specific_client = _detect_specific_client(user_query) if intent["client_query"] else None

    PAYLOAD_FIELDS = PayloadSelectorInclude(include=[
        "text", "file_name", "file_path", "file_url",
        "doc_type", "industry_vertical", "section_type",
        "source_folder", "page_numbers", "client_name",
    ])

    def _search(qdrant_filter):
        return qdrant_client().query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=RETRIEVAL_TOP_K,
            query_filter=qdrant_filter,
            with_payload=PAYLOAD_FIELDS,
        ).points

    # Build filter conditions from detected intent
    must = []

    if intent["client_query"]:
        if specific_client:
            # Direct lookup — filter by client_name so we always find that client's chunks
            must.append(FieldCondition(
                key="client_name",
                match=MatchValue(value=specific_client),
            ))
        else:
            must.append(FieldCondition(
                key="doc_type",
                match=MatchAny(any=["case_study", "solution_design", "proposal_client"]),
            ))

    if intent["roi_query"]:
        must.append(FieldCondition(
            key="has_roi_metrics",
            match=MatchValue(value=True),
        ))

    # Don't add vertical filter for specific client queries (client_name filter is enough)
    if intent["vertical"] and not specific_client:
        must.append(FieldCondition(
            key="industry_vertical",
            match=MatchValue(value=intent["vertical"]),
        ))

    qdrant_filter = Filter(must=must) if must else None
    results       = _search(qdrant_filter)

    # For specific client: no fallback — if client_name not in DB, return nothing
    if specific_client and len(results) < FALLBACK_THRESHOLD:
        return []

    # Fallback 1: drop vertical filter if too few results
    if len(results) < FALLBACK_THRESHOLD and intent["vertical"] and not specific_client and len(must) > 1:
        must_no_vertical = [c for c in must if getattr(c, "key", None) != "industry_vertical"]
        results = _search(Filter(must=must_no_vertical) if must_no_vertical else None)

    # Fallback 2: drop all filters only if truly empty (not for specific client queries)
    if len(results) < FALLBACK_THRESHOLD and qdrant_filter and not specific_client:
        results = _search(None)

    if not results:
        return []

    cleaned            = _clean_query(user_query)
    filter_boilerplate = cleaned != user_query

    BOILERPLATE = (
        "mindmap digital is a",
        "about mindmap digital",
        "mindmap digital the art of digital transformation",
    )

    chunks = [
        r.payload for r in results
        if not filter_boilerplate
        or not r.payload.get("text", "").strip().lower().startswith(BOILERPLATE)
    ]

    if not chunks:
        return []

    # For general client list queries (no specific client): deduplicate by client_name
    # so every named client gets representation, not just the top-ranked one
    if intent["client_query"] and not specific_client:
        named: Dict[str, Dict] = {}
        for c in chunks:  # already in Qdrant relevance order
            name = c.get("client_name")
            if name and name not in named:
                named[name] = c
        if not named:
            return []
        return list(named.values())

    # Normal Cohere reranking for specific client or non-client queries
    try:
        rerank_resp = cohere_client().rerank(
            model=RERANK_MODEL,
            query=_clean_query(user_query),
            documents=[c.get("text", "") for c in chunks],
            top_n=RERANK_TOP_N,
        )
        reranked = []
        for hit in rerank_resp.results:
            chunk = chunks[hit.index]
            chunk["_rerank_score"] = round(hit.relevance_score, 4)
            reranked.append(chunk)
    except Exception:
        # Cohere unavailable (rate limit / network) — fall back to Qdrant order
        reranked = chunks[:RERANK_TOP_N]

    # For specific client: verify the named client is actually present
    if specific_client:
        if not any(c.get("client_name") for c in reranked):
            return []

    return reranked


# ── Answer generation ─────────────────────────────────────────────────────────

ANSWER_SYSTEM = """
You are a sales assistant for MindMap Digital — an RPA and AI automation consultancy.
You help the internal sales team find relevant case studies, capabilities, ROI metrics,
and use cases from MindMap's sales collateral.

CRITICAL RULE — NO HALLUCINATION:
- You MUST only use information that appears word-for-word in the provided context chunks.
- If a client name, metric, percentage, or outcome is not explicitly stated in the context, DO NOT include it.
- Do NOT invent, infer, or generalise any facts, numbers, client names, or outcomes.
- If the context does not contain enough information to answer, say exactly:
  "No specific data available in current collateral. Check with the delivery team."

CLIENT NAMES — STRICT RULE:
- When asked about clients, ONLY list names that appear in the "Client:" field of the context chunks.
- NEVER infer or extract client names from the text content itself.
- If no chunk has a "Client:" field with a real value (i.e. all show "N/A"), say:
  "No specific client data available in current collateral. Check with the delivery team."

RESPONSE LENGTH:
- DEFAULT: Short — maximum 3 to 4 bullet points or 2 to 3 sentences.
- IN-DEPTH ONLY when the user explicitly says "explain in detail", "deep dive",
  "elaborate", "tell me more", or similar.

Other rules:
- Use bullet points or short sentences — no markdown headers (no #, ##, ###)
- Do not use emojis or informal language
- Do not include source citations or document references in your answer
- When listing ROI metrics, quote them exactly as they appear in the context
"""


def _format_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        client = chunk.get("client_name") or "N/A"
        parts.append(
            f"[Chunk {i}]\n"
            f"File: {chunk.get('file_name', 'Unknown')}\n"
            f"Type: {chunk.get('doc_type', 'Unknown')}\n"
            f"Client: {client}\n"
            f"Section: {chunk.get('section_type', 'Unknown')}\n"
            f"Verticals: {', '.join(chunk.get('industry_vertical', []))}\n\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)


def get_sources(chunks: List[Dict]) -> List[Dict]:
    """Extract unique source files from retrieved chunks, preserving rerank order."""
    seen: dict = {}
    for chunk in chunks:
        fname = chunk.get("file_name", "")
        if not fname:
            continue
        if fname not in seen:
            seen[fname] = {
                "file_name":         fname,
                "file_path":         chunk.get("file_path", ""),
                "file_url":          chunk.get("file_url", ""),
                "doc_type":          chunk.get("doc_type", ""),
                "industry_vertical": chunk.get("industry_vertical", []),
                "source_folder":     chunk.get("source_folder", ""),
                "page_numbers":      set(),
            }
        seen[fname]["page_numbers"].update(chunk.get("page_numbers", []))

    sources = []
    for src in seen.values():
        src["page_numbers"] = sorted(src["page_numbers"])
        sources.append(src)
    return sources


# ── Streaming ─────────────────────────────────────────────────────────────────

_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
}

_GREETING_RESPONSE = (
    "Hello! How can I help you with MindMap Digital's sales collateral? "
    "You can ask about case studies, capabilities, ROI metrics, or specific industry use cases."
)

_NO_DATA_RESPONSE = (
    "No specific data available in current collateral. Check with the delivery team."
)


def stream_answer(
    user_query: str,
    chunks: List[Dict],
    conversation_history: List[Dict],
) -> Generator[str, None, None]:
    """Stream an answer given pre-retrieved chunks (no retrieval inside)."""

    if user_query.strip().lower().rstrip("!.,") in _GREETINGS:
        yield _GREETING_RESPONSE
        return

    if not chunks:
        yield _NO_DATA_RESPONSE
        return

    context = _format_context(chunks)

    # Build explicit client-name allowlist so GPT-4o cannot extract names from text body
    actual_clients = sorted({c.get("client_name") for c in chunks if c.get("client_name")})
    client_note = (
        f"\n\nVALID CLIENT NAMES (from 'Client:' fields only): {', '.join(actual_clients)}. "
        "Do NOT mention any other client or company names found in the text. "
        "If the question asks which clients or companies we have worked with, "
        "list ONLY these names as the answer — do not say 'no data'."
    ) if actual_clients else ""

    messages = [{"role": "system", "content": ANSWER_SYSTEM}]
    messages.extend(conversation_history[-10:])
    messages.append({
        "role":    "user",
        "content": f"Context from sales collateral:\n\n{context}\n\nQuestion: {user_query}{client_note}",
    })

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
    """2-stage pipeline: retrieve + rerank → stream answer. (Used by CLI.)"""

    if user_query.strip().lower().rstrip("!.,") in _GREETINGS:
        yield _GREETING_RESPONSE
        return

    chunks = retrieve(user_query)
    yield from stream_answer(user_query, chunks, conversation_history)


# ── CLI ───────────────────────────────────────────────────────────────────────

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

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": full_response})
        if len(history) > 10:
            history = history[-10:]


if __name__ == "__main__":
    chat_cli()
