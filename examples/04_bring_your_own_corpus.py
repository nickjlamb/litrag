"""Point the pipeline at your own abstracts file.

The corpus format is plain markdown — one block per abstract:

    ## PMID: 12345678
    **Title:** The trial's title
    **Source:** N Engl J Med. 2024

    The abstract text...

This example writes a tiny two-abstract corpus to a temp file, ingests it,
builds a fresh (in-memory) index, and runs a retrieval query against it.
No API key required.

    python examples/04_bring_your_own_corpus.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from index import build_index  # noqa: E402
from ingest import chunk, load_corpus  # noqa: E402

MINI_CORPUS = """\
## PMID: 33567185
**Title:** Once-Weekly Semaglutide in Adults with Overweight or Obesity
**Source:** N Engl J Med. 2021

In participants with overweight or obesity, the mean change in body weight from
baseline to week 68 was -14.9% in the semaglutide group as compared with -2.4%
with placebo. Nausea and diarrhea were the most common adverse events.

## PMID: 37952131
**Title:** Semaglutide and Cardiovascular Outcomes in Obesity without Diabetes
**Source:** N Engl J Med. 2023

A primary cardiovascular end-point event occurred in 6.5% of the semaglutide
group and 8.0% of the placebo group (hazard ratio, 0.80; 95% confidence
interval, 0.72 to 0.90; P<0.001).
"""


def main() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(MINI_CORPUS)
        corpus_path = fh.name

    records = load_corpus(corpus_path)
    docs = chunk(records)
    print(f"Ingested {len(records)} abstracts -> {len(docs)} passages\n")

    store = build_index(docs)  # in-memory; use save_index()/get_or_build_index() to persist
    for doc in store.similarity_search("How much weight loss?", k=1):
        print(f"Top hit: [PMID {doc.metadata['pmid']}] {doc.metadata['title']}")
        print(f"  {doc.page_content[:160]}...")

    Path(corpus_path).unlink()


if __name__ == "__main__":
    main()
