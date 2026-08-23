# Changelog

All notable changes to LitRAG are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-23

First tagged release.

### Added

- **Ingestion** (`ingest.py`): markdown corpus parser (`## PMID:` / `**Title:**` / `**Source:**` blocks) and word-window chunking that carries `{pmid, title, source}` metadata through the whole pipeline.
- **Indexing** (`index.py`): local FAISS vector store over Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings, with build/save/load/get-or-build helpers — no managed vector-DB key required.
- **RAG chain** (`rag.py`): LangChain LCEL retrieval + generation chain returning a structured `{answer, claims:[{text, cited_quote, source}]}` — every claim must carry a verbatim quote from a retrieved passage.
- **Citation-faithfulness eval** (`faithfulness.py`): two-stage groundedness check — a deterministic quote locator (normalize + `rapidfuzz.partial_ratio`, 90-point threshold) that short-circuits fabricated quotes to `hallucinated_quote`, then an LLM-as-judge (forced tool use, prompt-cached passages) grading `supports` / `partial` / `contradicts` / `not_found`.
- **Demo** (`demo.py`): end-to-end run over three questions — two answerable from the corpus, one deliberately not — printing per-claim verdicts and an overall grounded flag.
- **Corpus** (`data/`): 15 real semaglutide abstracts (STEP, SELECT, SUSTAIN and related trials) so retrieval runs key-free out of the box.
- **Tests**: offline unit tests for the quote locator (exact hit, fuzzy hit, paraphrase rejection, fabricated quote, empty quote).
- **Examples** (`examples/`): four focused scripts — retrieval-only, ask-with-citations, catch-a-hallucination, bring-your-own-corpus.
- **Docs**: README with architecture diagram and quick start; from-the-build LangChain vs LlamaIndex framework notes (`docs/framework-notes.md`); contributing guide; issue and PR templates; `CITATION.cff`.
- **CI**: GitHub Actions workflow running `pytest` and `ruff check` on Python 3.10, 3.11, and 3.12.

[Unreleased]: https://github.com/nickjlamb/litrag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nickjlamb/litrag/releases/tag/v0.1.0
