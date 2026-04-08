from __future__ import annotations
import os
import re
from typing import Generator, List, Dict, Optional

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

COHERE_API_KEY  = os.getenv("COHERE_API_KEY")
RERANK_MODEL    = "rerank-english-v3.0"
RERANK_TOP_N    = 5
RETRIEVAL_TOP_K = 50

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
    "BFSI":         ["bfsi", "bank", "banking", "insurance", "fintech", "mortgage", "financial services"],
    "Pharma":       ["pharma", "pharmaceutical", "drug manufacturing"],
    "Telecom":      ["telecom", "telecommunications"],
    "Manufacturing":["manufacturing", "production", "plant", "factory"],
    "SCM":          ["supply chain", "demand planning", "procurement", "inventory"],
    "HR":           ["human resources", "payroll", "recruitment", "onboarding"],
    "FA":           ["accounts payable", "accounts receivable", "invoice", "fp&a", "financial planning"],
    "Government":   ["government", "public sector"],
    "Logistics":    ["logistics", "transportation", "freight"],
    "Aviation":     ["aviation", "airline", "airport"],
    "Education":    ["education", "university", "higher education"],
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


# Primary vertical for each named client — used to filter list queries so that
# healthcare clients don't appear under BFSI just because their docs mention banks
_CLIENT_PRIMARY_VERTICAL: Dict[str, str] = {
    "Kotak":                        "BFSI",
    "UAE Bank":                     "BFSI",
    "Wio Bank":                     "BFSI",
    "Authbridge":                   "BFSI",
    "Zurich":                       "BFSI",
    "NGA HR":                       "HR",
    "Intas Pharmaceuticals":        "Pharma",
    "Piramal Pharma":               "Pharma",
    "TheDDCGroup":                  "FA",
    "Fellowship Village":           "Healthcare",
    "Parker":                       "Healthcare",
    "United Methodist Communities": "Healthcare",
    "Ingleside":                    "Healthcare",
    "Archcare":                     "Healthcare",
    "BlueTide":                     "IT",
    "CleverCruit":                  "HR",
}


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
    stopwords = {"what","does","do","is","are","the","a","an","tell","me","about","how","why","who","where","when","your"}
    meaningful = [w for w in cleaned.split() if w.lower() not in stopwords]
    return cleaned if len(meaningful) >= 2 else query


def retrieve(user_query: str) -> List[Dict]:
    """Qdrant search (with intent-based filters) → Cohere rerank."""
    from qdrant_client.models import (
        Filter, FieldCondition, MatchAny, MatchValue, PayloadSelectorInclude,
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

    if intent["roi_query"] and not specific_client:
        must.append(FieldCondition(
            key="has_roi_metrics",
            match=MatchValue(value=True),
        ))
        # Exclude generic industry packs — prefer case studies and client-specific docs
        must.append(FieldCondition(
            key="doc_type",
            match=MatchAny(any=["case_study", "proposal_client", "assessment_sample", "solution_design"]),
        ))

    # Don't add vertical filter for specific client queries (client_name filter is enough)
    if intent["vertical"] and not specific_client:
        must.append(FieldCondition(
            key="industry_vertical",
            match=MatchValue(value=intent["vertical"]),
        ))

    qdrant_filter = Filter(must=must) if must else None
    results = _search(qdrant_filter)

    FALLBACK_THRESHOLD = 2

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

    # For general client list queries (no specific client): search directly by
    # client_name so doc_type restrictions don't hide named clients (e.g. Kotak
    # is doc_type=industry_pack which would be filtered out by the doc_type check)
    if intent["client_query"] and not specific_client:
        all_client_names = list(set(CLIENT_NAME_MAP.values()))
        client_must = [FieldCondition(
            key="client_name",
            match=MatchAny(any=all_client_names),
        )]
        if intent["vertical"]:
            client_must.append(FieldCondition(
                key="industry_vertical",
                match=MatchValue(value=intent["vertical"]),
            ))
        # Use high limit so small clients (e.g. UAE Bank with 2 chunks) aren't crowded out
        def _search_all(qdrant_filter):
            return qdrant_client().query_points(
                collection_name=COLLECTION_NAME,
                query=query_vec,
                limit=500,
                query_filter=qdrant_filter,
                with_payload=PAYLOAD_FIELDS,
            ).points

        client_results = _search_all(Filter(must=client_must))
        # Fallback: drop vertical filter if too few named-client results
        if len(client_results) < FALLBACK_THRESHOLD and intent["vertical"]:
            client_results = _search_all(Filter(must=client_must[:1]))
        named: Dict[str, Dict] = {}
        for r in client_results:
            name = r.payload.get("client_name")
            if not name or name in named:
                continue
            # If filtering by vertical, only include clients whose primary vertical matches
            if intent["vertical"]:
                primary = _CLIENT_PRIMARY_VERTICAL.get(name)
                if primary and primary != intent["vertical"]:
                    continue
            named[name] = r.payload
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


ANSWER_SYSTEM = """
You are a sales assistant for MindMap Digital — an RPA and AI automation consultancy.
You help the internal sales team find relevant case studies, capabilities, ROI metrics, and use cases from MindMap's sales collateral.

CRITICAL RULE — NO HALLUCINATION:
- You MUST only use information that appears word-for-word in the provided context chunks.
- If a client name, metric, percentage, or outcome is not explicitly stated in the context, DO NOT include it.
- Do NOT invent, infer, or generalise any facts, numbers, client names, or outcomes.
- If the context does not contain enough information to answer, say exactly: "No specific data available in current collateral. Check with the delivery team."

CLIENT NAMES — STRICT RULE:
- When asked about clients, ONLY list names that appear in the "Client:" field of the context chunks.
- NEVER infer or extract client names from the text content itself.
- If no chunk has a "Client:" field with a real value (i.e. all show "N/A"), say: "No specific client data available in current collateral. Check with the delivery team."

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


_GREETINGS = {"hi", "hello", "hey", "hiya", "howdy", "greetings", "good morning", "good afternoon", "good evening"}


def stream_answer(
    user_query: str,
    chunks: List[Dict],
    conversation_history: List[Dict],
) -> Generator[str, None, None]:
    """Stream an answer given pre-retrieved chunks (no retrieval inside)."""

    if user_query.strip().lower().rstrip("!.,") in _GREETINGS:
        yield "Hello! How can I help you with MindMap Digital's sales collateral? You can ask about case studies, capabilities, ROI metrics, or specific industry use cases."
        return

    if not chunks:
        yield "No specific data available in current collateral. Check with the delivery team."
        return

    context = _format_context(chunks)

    # Build an explicit client-name allowlist so GPT-4o cannot extract names from text body
    actual_clients = sorted({c.get("client_name") for c in chunks if c.get("client_name")})

    messages = [{"role": "system", "content": ANSWER_SYSTEM}]

    # Inject client constraint as a system message so it overrides conversation history
    if actual_clients:
        messages.append({"role": "system", "content": (
            f"OVERRIDE: For this query the ONLY valid client names are: {', '.join(actual_clients)}. "
            "Ignore any client names from earlier in the conversation. "
            "If asked which clients we worked with, list ONLY these names. "
            "Do NOT mention any other company names from the text."
        )})

    messages.extend(conversation_history[-10:])
    messages.append({
        "role": "user",
        "content": f"Context from sales collateral:\n\n{context}\n\nQuestion: {user_query}",
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


def generate_title(user_query: str, assistant_response: str) -> str:
    """Generate a short chat title (5-7 words) from the first Q&A exchange."""
    resp = openai_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate very short chat titles (4-6 words max). "
                    "Given a question and answer, output only the title — no quotes, no punctuation at the end."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {user_query}\n\nAnswer: {assistant_response[:300]}",
            },
        ],
        temperature=0.3,
        max_tokens=20,
    )
    return resp.choices[0].message.content.strip()


def generate_summary(messages: List[Dict]) -> str:
    """Generate a short 2-3 sentence summary of a conversation."""
    if not messages:
        return ""

    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )

    resp = openai_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarise sales assistant conversations in 2-3 sentences max. "
                    "Cover what was asked and the key facts or metrics found. "
                    "Be factual and concise. No bullet points, no headers."
                ),
            },
            {
                "role": "user",
                "content": f"Summarise this conversation:\n\n{transcript}",
            },
        ],
        temperature=0.3,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


def answer(
    user_query: str,
    conversation_history: List[Dict],
) -> Generator[str, None, None]:
    """2-stage pipeline: retrieve + rerank → stream answer. (Used by CLI.)"""

    if user_query.strip().lower().rstrip("!.,") in _GREETINGS:
        yield "Hello! How can I help you with MindMap Digital's sales collateral? You can ask about case studies, capabilities, ROI metrics, or specific industry use cases."
        return

    chunks = retrieve(user_query)
    yield from stream_answer(user_query, chunks, conversation_history)


# ═════════════════════════════════════════════════════════════════════════════
# CLI — interactive chat loop
# ═════════════════════════════════════════════════════════════════════════════

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
