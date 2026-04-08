"""
Check how many Qdrant points actually have file_url set.
Run: python check_urls.py
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, IsEmptyCondition
from config import COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY

client = QdrantClient(url=QDRANT_URL, **({"api_key": QDRANT_API_KEY} if QDRANT_API_KEY else {}))

# Scroll all points and check file_url
offset = None
has_url = {}
no_url = {}

while True:
    result, offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        offset=offset,
        with_payload=["file_name", "file_url"],
    )
    for point in result:
        fname = point.payload.get("file_name", "")
        url = point.payload.get("file_url", "")
        if url:
            has_url[fname] = url
        else:
            no_url[fname] = True
    if offset is None:
        break

print(f"\n✅ Files WITH file_url: {len(has_url)}")
for f, u in sorted(has_url.items()):
    print(f"  {f}: {u[:60]}...")

print(f"\n❌ Files WITHOUT file_url: {len(no_url)}")
for f in sorted(no_url.keys()):
    print(f"  {f}")
