"""Quick check: show doc_type and industry_vertical for specific clients."""
from qdrant_client import QdrantClient
from config import COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

for name in ("Kotak", "UAE Bank"):
    results, _ = client.scroll(
        COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="client_name", match=MatchValue(value=name))]),
        limit=3,
        with_payload=["client_name", "doc_type", "industry_vertical", "file_name"],
        with_vectors=False,
    )
    print(f"\n=== {name} ({len(results)} sampled) ===")
    for r in results:
        p = r.payload
        print(f"  file      : {p.get('file_name')}")
        print(f"  doc_type  : {p.get('doc_type')}")
        print(f"  verticals : {p.get('industry_vertical')}")
