"""Load the abstract corpus and chunk it into passages with source metadata.

Each passage carries {pmid, title, source} so retrieved text can be cited back
to its origin — the metadata the faithfulness eval depends on.

The static corpus lives in ``data/sample_abstracts.md`` (see that file for format).
``load_from_pubcrawl`` is an optional path that pulls fresh abstracts from PubMed
instead of the static sample — it mirrors what the PubCrawl MCP server's
``search_pubmed`` + ``get_abstract`` tools return, fetched here via NCBI E-utilities
so the code path is runnable without an MCP client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

DEFAULT_CORPUS = "data/sample_abstracts.md"

# One record block in the markdown corpus: a "## PMID: <id>" header, a **Title:** line,
# a **Source:** line, then the abstract body up to the next header.
_RECORD_RE = re.compile(
    r"^##\s*PMID:\s*(?P<pmid>\S+)\s*$"
    r"(?P<body>.*?)"
    r"(?=^##\s*PMID:|\Z)",
    re.MULTILINE | re.DOTALL,
)
_TITLE_RE = re.compile(r"^\*\*Title:\*\*\s*(?P<title>.+?)\s*$", re.MULTILINE)
_SOURCE_RE = re.compile(r"^\*\*Source:\*\*\s*(?P<source>.+?)\s*$", re.MULTILINE)


@dataclass
class Record:
    """A single abstract parsed out of the corpus, before chunking."""

    pmid: str
    title: str
    source: str
    text: str


def load_corpus(path: str = DEFAULT_CORPUS) -> list[Record]:
    """Read the abstract corpus from disk into plain records.

    Parses the ``## PMID:`` / ``**Title:**`` / ``**Source:**`` markdown layout into
    one :class:`Record` per abstract. Any prose before the first ``## PMID:`` header
    (the file's preamble) is ignored.
    """
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()

    records: list[Record] = []
    for match in _RECORD_RE.finditer(raw):
        body = match.group("body")
        title_m = _TITLE_RE.search(body)
        source_m = _SOURCE_RE.search(body)

        # The abstract text is everything after the metadata lines.
        text = body
        for m in (title_m, source_m):
            if m:
                text = text.replace(m.group(0), "", 1)
        text = text.strip()

        records.append(
            Record(
                pmid=match.group("pmid").strip(),
                title=title_m.group("title").strip() if title_m else "(untitled)",
                source=source_m.group("source").strip() if source_m else "PubMed",
                text=text,
            )
        )
    return records


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word windows.

    We measure passage size in words as a cheap proxy for tokens (~1 word ≈ 1.3
    tokens for English prose). Abstracts are short, so most fall into a single
    passage — which keeps each retrieved chunk cleanly attributable to one source.
    """
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []

    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def chunk(records: list[Record], size: int = 512, overlap: int = 64) -> list[Document]:
    """Chunk records into passages, preserving {pmid, title, source} metadata.

    ``size`` and ``overlap`` are in words (token proxy). Returns LangChain
    :class:`~langchain_core.documents.Document` objects ready for embedding.
    """
    docs: list[Document] = []
    for rec in records:
        pieces = _split_words(rec.text, size, overlap)
        for i, piece in enumerate(pieces):
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "pmid": rec.pmid,
                        "title": rec.title,
                        "source": rec.source,
                        "chunk": i,
                    },
                )
            )
    return docs


def load_documents(path: str = DEFAULT_CORPUS, size: int = 512, overlap: int = 64) -> list[Document]:
    """Convenience: load the corpus and chunk it in one call."""
    return chunk(load_corpus(path), size=size, overlap=overlap)


def load_from_pubcrawl(query: str, max_results: int = 15) -> list[Record]:
    """Pull fresh abstracts from PubMed for ``query`` (optional ``--from-pubcrawl`` path).

    Mirrors the PubCrawl MCP server's ``search_pubmed`` + ``get_abstract`` flow, but
    fetched directly from NCBI E-utilities so it runs without an MCP client. Returns
    the same :class:`Record` shape as :func:`load_corpus`, so the rest of the pipeline
    is identical whether the corpus is static or freshly pulled.
    """
    import json
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def _get(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted host)
            return resp.read()

    # 1. esearch -> PMIDs
    search_url = (
        f"{eutils}/esearch.fcgi?db=pubmed&retmode=json&retmax={max_results}"
        f"&term={urllib.parse.quote(query)}"
    )
    pmids = json.loads(_get(search_url))["esearchresult"]["idlist"]
    if not pmids:
        return []

    # 2. efetch -> abstract XML for all PMIDs at once
    fetch_url = f"{eutils}/efetch.fcgi?db=pubmed&retmode=xml&id={','.join(pmids)}"
    root = ET.fromstring(_get(fetch_url))

    records: list[Record] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        journal_el = article.find(".//Journal/ISOAbbreviation")
        year_el = article.find(".//JournalIssue/PubDate/Year")

        # Abstract may be split into labelled sections; join them with their labels.
        parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            text = "".join(ab.itertext()).strip()
            parts.append(f"{label}: {text}" if label else text)
        abstract = "\n\n".join(p for p in parts if p)
        if not abstract:
            continue

        journal = journal_el.text if journal_el is not None else "PubMed"
        year = year_el.text if year_el is not None else ""
        source = f"PubMed — {journal}{', ' + year if year else ''}".strip()

        records.append(
            Record(
                pmid=pmid_el.text if pmid_el is not None else "",
                title="".join(title_el.itertext()).strip()
                if title_el is not None
                else "(untitled)",
                source=source,
                text=abstract,
            )
        )
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and chunk the abstract corpus.")
    parser.add_argument("--path", default=DEFAULT_CORPUS, help="corpus markdown file")
    parser.add_argument("--from-pubcrawl", metavar="QUERY", help="pull fresh abstracts from PubMed instead")
    parser.add_argument("--size", type=int, default=512, help="passage size in words")
    parser.add_argument("--overlap", type=int, default=64, help="passage overlap in words")
    args = parser.parse_args()

    if args.from_pubcrawl:
        records = load_from_pubcrawl(args.from_pubcrawl)
        print(f"Pulled {len(records)} abstracts from PubMed for {args.from_pubcrawl!r}")
    else:
        records = load_corpus(args.path)
        print(f"Loaded {len(records)} abstracts from {args.path}")

    docs = chunk(records, size=args.size, overlap=args.overlap)
    print(f"Chunked into {len(docs)} passages.\n")
    for d in docs[:3]:
        print(f"[{d.metadata['pmid']}] {d.metadata['title'][:70]}")
        print(f"  {d.page_content[:120]}...\n")
