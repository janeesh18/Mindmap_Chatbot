"""
Debug what chunks are retrieved for a query.
Run: python debug_query.py
"""
from rag import retrieve

query = "Jubilant project"
chunks = retrieve(query)

print(f"\n{len(chunks)} chunks retrieved:\n")
for i, chunk in enumerate(chunks, 1):
    print(f"{'='*60}")
    print(f"[{i}] {chunk.get('file_name')} | score: {chunk.get('_rerank_score')}")
    print(f"    Type: {chunk.get('doc_type')} | Section: {chunk.get('section_type')}")
    print(f"    Text: {chunk.get('text', '')[:400]}")
    print()
