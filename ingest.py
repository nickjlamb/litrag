"""Load the abstract corpus and chunk it into passages with source metadata.

Each passage carries {pmid, title, source} so retrieved text can be cited back
to its origin — the metadata the faithfulness eval depends on.

TODO (build session):
- load_corpus(path): read data/sample_abstracts.* into records
- chunk(records, size=512, overlap=64): -> list[langchain_core.documents.Document]
  with metadata={"pmid", "title", "source"}
- (optional) load_from_pubcrawl(query): pull fresh abstracts via the PubCrawl MCP
  server instead of the static sample
"""

from __future__ import annotations


def load_corpus(path: str):
    """Read the abstract corpus from disk into plain records."""
    raise NotImplementedError


def chunk(records, size: int = 512, overlap: int = 64):
    """Chunk records into passages, preserving {pmid, title, source} metadata."""
    raise NotImplementedError
