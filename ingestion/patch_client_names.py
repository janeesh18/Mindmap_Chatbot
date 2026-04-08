"""
Patch existing Qdrant points with client_name field.
Run once after adding CLIENT_NAME_MAP to config — no re-ingestion needed.

Usage:
    python patch_client_names.py
    python patch_client_names.py --dry-run   # preview only, no writes
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import SetPayload, PointIdsList

from config import CLIENT_NAME_MAP, COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY


def detect_client(file_name: str) -> str | None:
    searchable = file_name.lower()
    for key, name in CLIENT_NAME_MAP.items():
        if key in searchable:
            return name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

    print(f"\nScrolling all points in '{COLLECTION_NAME}' …")
    all_points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            COLLECTION_NAME, limit=500, offset=offset,
            with_payload=["file_name", "client_name"], with_vectors=False,
        )
        all_points.extend(batch)
        if offset is None:
            break

    print(f"Total points : {len(all_points)}")

    # Group points by detected client name
    to_update: dict[str, list] = {}   # client_name → [point_ids]
    already_set = 0

    for pt in all_points:
        existing = pt.payload.get("client_name")
        if existing:
            already_set += 1
            continue
        file_name  = pt.payload.get("file_name", "")
        name       = detect_client(file_name)
        if name:
            to_update.setdefault(name, []).append(pt.id)

    print(f"Already have client_name : {already_set}")
    print(f"Points to update         : {sum(len(v) for v in to_update.values())}")
    print(f"Points with no match     : {len(all_points) - already_set - sum(len(v) for v in to_update.values())}")
    print()

    counts = Counter()
    for client_name, ids in sorted(to_update.items()):
        print(f"  {client_name:<35} {len(ids)} chunks")
        counts[client_name] = len(ids)

    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
        return

    print("\nApplying patches …")
    BATCH = 200
    for client_name, ids in to_update.items():
        for i in range(0, len(ids), BATCH):
            batch_ids = ids[i: i + BATCH]
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"client_name": client_name},
                points=PointIdsList(points=batch_ids),
            )
        print(f"  OK {client_name} - {len(ids)} points updated")

    print(f"\nDone. {sum(counts.values())} points patched.\n")


if __name__ == "__main__":
    main()
