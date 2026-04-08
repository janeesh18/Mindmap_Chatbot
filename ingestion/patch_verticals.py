"""
Patch existing Qdrant points with missing industry_vertical values.
Run once to ensure all named-client chunks have the correct vertical tagged.

Usage:
    python patch_verticals.py
    python patch_verticals.py --dry-run   # preview only, no writes
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import SetPayload, PointIdsList

from config import COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY

# Maps canonical client_name → vertical(s) to ensure are present
CLIENT_VERTICAL_MAP: dict[str, list[str]] = {
    "Kotak":                    ["BFSI"],
    "UAE Bank":                 ["BFSI"],
    "Wio Bank":                 ["BFSI"],
    "Authbridge":               ["BFSI"],
    "Zurich":                   ["BFSI"],
    "NGA HR":                   ["HR"],
    "Intas Pharmaceuticals":    ["Healthcare"],
    "Piramal Pharma":           ["Healthcare"],
    "TheDDCGroup":              ["FA"],
    "Fellowship Village":       ["Healthcare"],
    "Parker":                   ["Healthcare"],
    "United Methodist Communities": ["Healthcare"],
    "Ingleside":                ["Healthcare"],
    "Archcare":                 ["Healthcare"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

    print(f"\nScrolling all points in '{COLLECTION_NAME}' ...")
    all_points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            COLLECTION_NAME, limit=500, offset=offset,
            with_payload=["client_name", "industry_vertical"], with_vectors=False,
        )
        all_points.extend(batch)
        if offset is None:
            break

    print(f"Total points: {len(all_points)}")

    # Find points that need vertical patching
    to_patch: dict[str, list] = defaultdict(list)  # vertical_key → point_ids

    for pt in all_points:
        client_name = pt.payload.get("client_name")
        if not client_name or client_name not in CLIENT_VERTICAL_MAP:
            continue
        required_verticals = CLIENT_VERTICAL_MAP[client_name]
        existing_verticals = pt.payload.get("industry_vertical") or []
        missing = [v for v in required_verticals if v not in existing_verticals]
        if missing:
            # Key by the new vertical list we need to set (add missing to existing)
            new_verticals = sorted(set(existing_verticals) | set(required_verticals))
            key = "|".join(new_verticals)
            to_patch[key].append((pt.id, new_verticals))

    total = sum(len(v) for v in to_patch.values())
    print(f"Points needing vertical patch: {total}\n")

    if total == 0:
        print("Nothing to patch.")
        return

    # Show preview grouped by client
    from collections import Counter
    preview: Counter = Counter()
    for key, entries in to_patch.items():
        verticals_label = key.replace("|", ", ")
        print(f"  vertical={verticals_label:<30} {len(entries)} points")

    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
        return

    print("\nApplying patches ...")
    BATCH = 200
    patched = 0
    for key, entries in to_patch.items():
        new_verticals = entries[0][1]
        ids = [e[0] for e in entries]
        for i in range(0, len(ids), BATCH):
            batch_ids = ids[i: i + BATCH]
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"industry_vertical": new_verticals},
                points=PointIdsList(points=batch_ids),
            )
        patched += len(ids)
        print(f"  OK vertical={new_verticals} - {len(ids)} points")

    print(f"\nDone. {patched} points patched.\n")


if __name__ == "__main__":
    main()
