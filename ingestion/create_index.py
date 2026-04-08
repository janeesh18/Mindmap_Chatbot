from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from config import COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY

client = QdrantClient(url=QDRANT_URL, **({"api_key": QDRANT_API_KEY} if QDRANT_API_KEY else {}))

for field in ("file_name", "client_name"):
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field,
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"Index created: {field}")

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="has_roi_metrics",
    field_schema=PayloadSchemaType.BOOL,
)
print("Index created: has_roi_metrics")
