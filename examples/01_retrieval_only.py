"""Retrieval only — no API key required.

Builds (or loads) the FAISS index over the sample corpus and prints the top-k
passages for a query, with their provenance metadata. Everything here runs
locally: the only download is the ~90 MB sentence-transformers model on first run.

    python examples/01_retrieval_only.py "Does semaglutide reduce cardiovascular events?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from index import get_or_build_index  # noqa: E402
from ingest import load_documents  # noqa: E402

DEFAULT_QUERY = "Does semaglutide reduce cardiovascular events?"


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    print("Building/loading FAISS index ...")
    store = get_or_build_index(load_documents())

    print(f"\nQuery: {query}\n")
    for rank, (doc, score) in enumerate(store.similarity_search_with_score(query, k=4), 1):
        meta = doc.metadata
        print(f"#{rank}  [PMID {meta['pmid']}]  distance={score:.3f}")
        print(f"    {meta['title']}")
        print(f"    {doc.page_content[:200]}...\n")


if __name__ == "__main__":
    main()
