from __future__ import annotations
import json
import os
from typing import Generator, List, Dict, Optional

import cohere
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

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
RERANK_MODEL   = "rerank-english-v3.0"
RERANK_TOP_N   = 5    
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
        _cohere = cohere.client.Client(api_key=COHERE_API_KEY)
    return _cohere

INTENT_SYSTEM = """
You are a query parser for a sales collateral chatbot at MindMap Digital — an RPA/AI automation consultancy.

Given a user query, extract a JSON object with these exact fields:

{
  "semantic_query": "cleaned, focused search query",
  "intent": one of [find_case_study, ask_capability, find_use_cases, ask_roi, find_proposal, general],
  "filters": {
    "industry_vertical": string or null,   // e.g. "Healthcare", "BFSI", "Manufacturing"
    "doc_type": string or null,            // e.g. "case_study", "capability_deck", "industry_pack", "proposal_client"
    "function_area": string or null,       // e.g. "AP", "HR", "ITSM", "FPA"
    "has_roi_metrics": boolean or null
  }
}

Industry verticals: BFSI, Healthcare, Pharma, FA, HR, IT, Manufacturing, SCM, Telecom, Logistics, Government, Retail, Education, SAP, Aviation, Utilities, General
Doc types: case_study, capability_deck, industry_pack, proposal_client, assessment_sample, heatmap
Function areas: AP, AR, FPA, HR, ITSM, Trade, SCM, Customer Service, Compliance, Clinical, GL/Accounting

Rules:
- Only set a filter if you are confident — wrong filters hurt retrieval
- If the query is vague or general, set all filters to null
- semantic_query should be clean and focused, removing filler words
- Return ONLY valid JSON, no markdown, no explanation
"""


def parse_query(user_query: str, conversation_history: List[Dict]) -> Dict:
    """Stage 1: Extract intent + filters from user query."""
    
    # Include last 2 turns for context-aware parsing
    recent = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
    messages = [{"role": "system", "content": INTENT_SYSTEM}]
    messages.extend(recent)
    messages.append({"role": "user", "content": user_query})

    resp = openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=300,
    )

    raw = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "semantic_query": user_query,
            "intent": "general",
            "filters": {
                "industry_vertical": None,
                "doc_type": None,
                "function_area": None,
                "has_roi_metrics": None,
            },
        }
    return parsed

INTENT_TOP_K = {
    "find_case_study": 20,
    "ask_roi":         20,
    "ask_capability":  15,
    "find_use_cases":  20,
    "find_proposal":   15,
    "general":         20,
}


def _build_qdrant_filter(filters: Dict) -> Optional[Filter]:
    """Build Qdrant payload filter from extracted metadata."""
    conditions = []

    if filters.get("industry_vertical"):
        conditions.append(
            FieldCondition(
                key="industry_vertical",
                match=MatchAny(any=[filters["industry_vertical"]]),
            )
        )

    if filters.get("doc_type"):
        conditions.append(
            FieldCondition(
                key="doc_type",
                match=MatchValue(value=filters["doc_type"]),
            )
        )

    if filters.get("function_area"):
        conditions.append(
            FieldCondition(
                key="function_area",
                match=MatchAny(any=[filters["function_area"]]),
            )
        )

    if filters.get("has_roi_metrics") is True:
        conditions.append(
            FieldCondition(
                key="has_roi_metrics",
                match=MatchValue(value=True),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def _embed_query(text: str) -> List[float]:
    resp = openai_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text.replace("\n", " ")],
    )
    return resp.data[0].embedding

def retrieve(semantic_query: str, filters: Dict, intent: str) -> List[Dict]:
    """
    Stage 2: Qdrant filtered search → Cohere rerank.
    Falls back to unfiltered search if filtered returns < 3 results.
    """
    top_k      = INTENT_TOP_K.get(intent, RETRIEVAL_TOP_K)
    query_vec  = _embed_query(semantic_query)
    qfilter    = _build_qdrant_filter(filters)
    
    result= qdrant_client().search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        query_filter=qfilter,
        limit=top_k,
        with_payload=True,
    )

    if len(results) < 3 and qfilter is not None:
        print(f"  [RETRIEVAL] Only {len(results)} filtered results — falling back to unfiltered")
        results = qdrant_client().search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )

    if not results:
        return []

    chunks = [r.payload for r in results]

    docs_for_rerank = [c.get("text", "") for c in chunks]

    rerank_resp = cohere_client().rerank(
        model=RERANK_MODEL,
        query=semantic_query,
        documents=docs_for_rerank,
        top_n=RERANK_TOP_N,
    )

    reranked = []
    for hit in rerank_resp.results:
        chunk = chunks[hit.index]
        chunk["_rerank_score"] = round(hit.relevance_score, 4)
        reranked.append(chunk)

    return reranked

ANSWER_SYSTEM = """
You are a helpful sales assistant for MindMap Digital — an RPA and AI automation consultancy.

You help the internal sales team find relevant case studies, capabilities, ROI metrics, and use cases from MindMap's sales collateral.

Rules:
- Answer only from the provided context chunks
- Be concise and direct — sales people are busy
- Always end your answer with source citations in this format:
  📄 [Document: {file_name} | Type: {doc_type} | Section: {section_type}]
- If multiple sources support the answer, cite all of them
- If the context doesn't have enough information, say: "I don't have specific data on that in our collateral. You may want to check with the delivery team."
- Never hallucinate metrics, client names, or outcomes
- When listing ROI metrics, quote them exactly as they appear in the source
"""


def _format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i}]\n"
            f"File: {chunk.get('file_name', 'Unknown')}\n"
            f"Type: {chunk.get('doc_type', 'Unknown')}\n"
            f"Section: {chunk.get('section_type', 'Unknown')}\n"
            f"Verticals: {', '.join(chunk.get('industry_vertical', []))}\n"
            f"Rerank Score: {chunk.get('_rerank_score', 'N/A')}\n\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)


def answer(
    user_query: str,
    conversation_history: List[Dict],
) -> Generator[str, None, None]:
    """
    Full 3-stage pipeline. Returns a streaming generator of text chunks.
    Usage:
        for token in answer("show me healthcare ROI case studies", history):
            print(token, end="", flush=True)
    """
    parsed   = parse_query(user_query, conversation_history)
    sem_q    = parsed.get("semantic_query", user_query)
    filters  = parsed.get("filters", {})
    intent   = parsed.get("intent", "general")

    print(f"\n[Query Parser] intent={intent} | filters={filters}")
    print(f"[Semantic Query] {sem_q}")

    chunks = retrieve(sem_q, filters, intent)

    if not chunks:
        yield "I don't have relevant information in our sales collateral for that query. Try rephrasing or ask about a specific industry or solution type."
        return

    print(f"[Retrieval] {len(chunks)} chunks after reranking")

    context = _format_context(chunks)

    messages = [{"role": "system", "content": ANSWER_SYSTEM}]

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
