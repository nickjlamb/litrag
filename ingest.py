"""Load the abstract corpus and chunk it into passages with source metadata.

Each passage carries {pmid, title, source} so retrieved text can be cited back
to its origin — the metadata the faithfulness eval depends on.

The static corpus lives in ``data/sample_abstracts.md`` (see that file for format).
``load_from_pubcrawl`` pulls fresh abstracts from PubMed instead of the static
sample. Its primary path talks to the PubCrawl MCP server
(https://www.pharmatools.ai/pubcrawl) over stdio via the official ``mcp`` client —
``search_pubmed`` for PMIDs, then ``get_abstract`` per article. A ``via="eutils"``
fallback hits NCBI E-utilities directly for environments without Node/npm.
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


# Unicode spaces PubMed HTML entities decode to (thin space, no-break space, etc.);
# normalized to plain spaces so quotes stay matchable by the faithfulness locator.
_ODD_SPACE_RE = re.compile("[\u2000-\u200a\u202f\u00a0]")  # thin/en/em spaces, nbsp


def _clean(text: str) -> str:
    """Decode HTML entities and normalize exotic whitespace to plain text."""
    import html

    return _ODD_SPACE_RE.sub(" ", html.unescape(text)).strip()


def _format_source(meta: dict) -> str:
    """Build a ``**Source:**`` line like ``PubMed — N Engl J Med, 2024;30(7):2058-2066.``"""
    cite = _clean(meta.get("journal") or "") or "PubMed"
    if meta.get("year"):
        cite += f", {meta['year']}"
    if meta.get("volume"):
        cite += f";{meta['volume']}"
        if meta.get("issue"):
            cite += f"({meta['issue']})"
        if meta.get("pages"):
            cite += f":{meta['pages']}"
    return f"PubMed — {cite}."


def load_from_pubcrawl(query: str, max_results: int = 15, via: str = "mcp") -> list[Record]:
    """Pull fresh abstracts from PubMed for ``query`` (the ``--from-pubcrawl`` path).

    ``via="mcp"`` (default) talks to the PubCrawl MCP server — one of our own tools —
    spawned locally over stdio (``npm install -g @pharmatools/pubcrawl``).
    ``via="eutils"`` hits NCBI E-utilities directly, for environments without Node.
    Both return the same :class:`Record` shape as :func:`load_corpus`, so the rest of
    the pipeline is identical whether the corpus is static or freshly pulled.
    """
    if via == "mcp":
        return _load_via_pubcrawl_mcp(query, max_results)
    if via == "eutils":
        return _load_via_eutils(query, max_results)
    raise ValueError(f"via must be 'mcp' or 'eutils', got {via!r}")


def _load_via_pubcrawl_mcp(query: str, max_results: int) -> list[Record]:
    """Fetch via the PubCrawl MCP server: ``search_pubmed`` then ``get_abstract``.

    NCBI rate-limits E-utilities (3 req/s without a key, 10 with), so transient
    429s are retried with exponential backoff. Set ``NCBI_API_KEY`` for the higher
    tier — it's forwarded to the spawned server, which passes it through to NCBI.
    """
    import asyncio
    import json
    import os
    import shutil

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client
    except ImportError as exc:
        raise RuntimeError(
            "The 'mcp' package is required for the PubCrawl MCP path: pip install mcp "
            "(or fall back with via='eutils')."
        ) from exc

    if shutil.which("pubcrawl") is None:
        raise RuntimeError(
            "PubCrawl MCP server not found on PATH. Install it with "
            "'npm install -g @pharmatools/pubcrawl' (no API key needed), "
            "or fall back with via='eutils'."
        )

    def _payload(result) -> dict:
        """First JSON text block of an MCP tool result.

        PubCrawl returns tool results as pretty-printed JSON in a text block;
        failures (e.g. no network to NCBI) come back as plain text — surface those.
        """
        for block in result.content:
            if getattr(block, "type", None) == "text":
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    raise RuntimeError(f"PubCrawl error: {block.text}") from None
        raise RuntimeError("PubCrawl returned no text content")

    async def _call(session, tool: str, args: dict, attempts: int = 4) -> dict:
        """Call a PubCrawl tool, retrying with backoff on NCBI 429 rate limits."""
        delay = 1.0
        for attempt in range(attempts):
            result = await session.call_tool(tool, args)
            try:
                return _payload(result)
            except RuntimeError as exc:
                if "429" in str(exc) and attempt < attempts - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise

    # The stdio client spawns the server with a minimal environment, so forward
    # NCBI_API_KEY explicitly (raises NCBI's limit from 3 to 10 req/s if set),
    # merged onto the safe defaults (PATH etc.) the server needs to start.
    env = get_default_environment()
    if os.environ.get("NCBI_API_KEY"):
        env["NCBI_API_KEY"] = os.environ["NCBI_API_KEY"]

    async def _run() -> list[Record]:
        params = StdioServerParameters(command="pubcrawl", env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                hits = (
                    await _call(session, "search_pubmed", {"query": query, "maxResults": max_results})
                ).get("results", [])

                records: list[Record] = []
                for hit in hits:
                    meta = await _call(session, "get_abstract", {"pmid": hit["pmid"]})

                    parts = []
                    for sec in meta.get("abstract_sections") or []:
                        text = _clean(sec.get("text", ""))
                        label = (sec.get("label") or "").strip()
                        if text:
                            parts.append(f"{label}: {text}" if label else text)
                    if not parts:
                        continue  # no abstract — nothing to ground claims in

                    records.append(
                        Record(
                            pmid=str(meta.get("pmid", hit["pmid"])),
                            title=_clean(meta.get("title") or "(untitled)"),
                            source=_format_source(meta),
                            text="\n\n".join(parts),
                        )
                    )
                return records

    try:
        return asyncio.run(_run())
    except BaseException as exc:
        # anyio task groups wrap failures in ExceptionGroups; unwrap single-error
        # groups (duck-typed for py310) so callers see the plain error, not a
        # nested group traceback.
        inner = exc
        while hasattr(inner, "exceptions") and len(inner.exceptions) == 1:
            inner = inner.exceptions[0]
        raise inner from None


def save_corpus(records: list[Record], path: str, heading: str | None = None) -> None:
    """Write records to ``path`` in the documented corpus format.

    The inverse of :func:`load_corpus`: a ``## PMID:`` / ``**Title:**`` /
    ``**Source:**`` block per abstract, so a freshly pulled corpus is a drop-in
    replacement for the static sample (``load_documents(path=...)``).
    """
    lines: list[str] = []
    if heading:
        lines += [f"# {heading}", ""]
    for rec in records:
        lines += [
            f"## PMID: {rec.pmid}",
            f"**Title:** {rec.title}",
            f"**Source:** {rec.source}",
            "",
            rec.text,
            "",
        ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _load_via_eutils(query: str, max_results: int) -> list[Record]:
    """Fallback: mirror the PubCrawl flow directly against NCBI E-utilities."""
    import json
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def _get(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=30) as resp:  # trusted host (NCBI)
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
    parser.add_argument(
        "--via",
        choices=["mcp", "eutils"],
        default="mcp",
        help="fetch through the PubCrawl MCP server (default) or NCBI E-utilities",
    )
    parser.add_argument("--max-results", type=int, default=15, help="abstracts to pull with --from-pubcrawl")
    parser.add_argument("--save", metavar="PATH", help="write pulled abstracts as a corpus file (with --from-pubcrawl)")
    parser.add_argument("--size", type=int, default=512, help="passage size in words")
    parser.add_argument("--overlap", type=int, default=64, help="passage overlap in words")
    args = parser.parse_args()

    if args.from_pubcrawl:
        records = load_from_pubcrawl(args.from_pubcrawl, max_results=args.max_results, via=args.via)
        origin = "the PubCrawl MCP server" if args.via == "mcp" else "NCBI E-utilities"
        print(f"Pulled {len(records)} abstracts via {origin} for {args.from_pubcrawl!r}")
        if args.save:
            save_corpus(records, args.save, heading=f"Corpus — {args.from_pubcrawl} (via PubCrawl)")
            print(f"Saved corpus to {args.save}")
    else:
        records = load_corpus(args.path)
        print(f"Loaded {len(records)} abstracts from {args.path}")

    docs = chunk(records, size=args.size, overlap=args.overlap)
    print(f"Chunked into {len(docs)} passages.\n")
    for d in docs[:3]:
        print(f"[{d.metadata['pmid']}] {d.metadata['title'][:70]}")
        print(f"  {d.page_content[:120]}...\n")
