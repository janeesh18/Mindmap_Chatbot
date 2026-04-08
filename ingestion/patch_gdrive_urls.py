"""
Patch Qdrant payloads with Google Drive download URLs.

Usage:
    1. Fill in gdrive_urls.json with filename → Google Drive share URL mappings.
    2. Run:  python patch_gdrive_urls.py

The script converts share URLs to direct download URLs and sets `file_url`
on every Qdrant point whose file_name matches an entry in gdrive_urls.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, SetPayload

from config import COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL

GDRIVE_URLS_FILE = Path(__file__).parent / "gdrive_urls.json"


def share_url_to_download_url(share_url: str) -> str:
    """Convert a Google Drive share URL to a direct download URL."""
    match = re.search(r"/file/d/([^/]+)", share_url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    # Already a direct download URL or unknown format — return as-is
    return share_url


def main() -> None:
    with open(GDRIVE_URLS_FILE, encoding="utf-8") as f:
        raw: dict = json.load(f)

    # Remove the instructions key
    url_map = {k: v for k, v in raw.items() if not k.startswith("_")}

    client = QdrantClient(
        url=QDRANT_URL,
        **({"api_key": QDRANT_API_KEY} if QDRANT_API_KEY else {}),
    )

    updated = 0
    for filename, share_url in url_map.items():
        download_url = share_url_to_download_url(share_url)

        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"file_url": download_url},
            points=Filter(
                must=[FieldCondition(key="file_name", match=MatchValue(value=filename))]
            ),
        )
        print(f"  [OK] {filename}")
        updated += 1

    print(f"\nDone — patched {updated} file(s) in Qdrant.")


if __name__ == "__main__":
    main()
