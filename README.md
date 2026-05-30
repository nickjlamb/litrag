# LitRAG

**Grounded retrieval-augmented generation over the medical literature — with a citation-faithfulness eval built in.**

Most RAG demos stop at "it retrieved something and wrote an answer." LitRAG goes one step further: it *checks whether the generated answer is actually supported by the retrieved sources*, and flags hallucinated or unsupported claims. That groundedness layer — not the pipeline — is the point.

Built on **LangChain** (orchestration) + **Hugging Face** `sentence-transformers` (embeddings) + **FAISS** (vector store), so it runs locally with no managed vector-DB key required.

> Status: scaffold. See the build plan below.

---

## Why this exists

Two things at once:

1. **A faithful RAG reference.** A small, readable pipeline that retrieves from PubMed abstracts and answers questions *with citations*, then verifies those citations hold up.
2. **An honest framework comparison.** The pipeline is implemented in LangChain; the README documents how the same retrieval would look in LlamaIndex, and where each framework's abstraction helps vs. gets in the way. (See [Framework notes](#framework-notes-langchain-vs-llamaindex).)

The groundedness eval reuses the citation-faithfulness approach from a separate cookbook notebook: locate the cited quote in the source deterministically, then use an LLM-as-judge to grade whether the source *supports* the claim (supports / partial / contradicts / not-found).

---

## Architecture

```
PubMed abstracts ──▶ chunk ──▶ HF sentence-transformers embeddings ──▶ FAISS index
                                                                          │
                              question ──▶ retrieve top-k ──────────────┘
                                              │
                                              ▼
                                   LangChain RAG chain (Claude / OpenAI)
                                              │
                                              ▼
                                   answer + cited passages
                                              │
                                              ▼
                              citation-faithfulness eval  ──▶  grounded? / flagged claims
```

## Layout

| File | Role |
|------|------|
| `ingest.py` | Load the abstract corpus, chunk into passages with source metadata |
| `index.py` | Build/load the FAISS index from HF `sentence-transformers` embeddings |
| `rag.py` | LangChain retrieval + generation chain; returns answer **with** cited passages |
| `faithfulness.py` | Citation-faithfulness eval — locate quote in source, LLM-judge support level |
| `demo.py` | End-to-end run: ingest → index → ask → answer → grade |
| `data/` | Sample PubMed abstracts (static sample so the repo runs key-free for retrieval) |

## Quickstart (after build)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY (or OPENAI_API_KEY) for the generation + judge steps
python demo.py
```

Embedding + retrieval run fully local (HF + FAISS); only generation and the faithfulness judge call an LLM API.

---

## Build plan (for the build session)

Implement in this order — each step is independently runnable:

1. **`ingest.py`** — load `data/sample_abstracts.*`, chunk to ~512-token passages, attach `{pmid, title, source}` metadata. (Optional: a `--from-pubcrawl` path that pulls fresh abstracts via the PubCrawl MCP server instead of the static sample.)
2. **`index.py`** — embed passages with `langchain_huggingface.HuggingFaceEmbeddings` (model e.g. `sentence-transformers/all-MiniLM-L6-v2`), build a `langchain_community.vectorstores.FAISS` index, save/load from disk.
3. **`rag.py`** — a LangChain retrieval chain (`chat model` = `langchain_anthropic.ChatAnthropic` with `claude-sonnet-4-6`, or `langchain_openai`), prompted to answer **and quote the supporting passage per claim**. Return structured `{answer, claims:[{text, cited_quote, source}]}`.
4. **`faithfulness.py`** — for each claim: locate `cited_quote` in the retrieved source (normalize + `rapidfuzz.partial_ratio`); if found, LLM-judge support level (supports/partial/contradicts/not-found). Short-circuit to "hallucinated quote" if the quote isn't locatable. (Port the logic from the cookbook `citation_faithfulness.py`.)
5. **`demo.py`** — wire it end-to-end on 2–3 example questions; print answer + per-claim grounded verdict.
6. **Tests** — a couple of unit tests on the quote-locator (exact hit, fuzzy hit, fabricated quote → not found).
7. **Fill in [Framework notes](#framework-notes-langchain-vs-llamaindex)** from the actual build experience — be specific about where LangChain's abstraction earned its keep and where it cost a layer of indirection.

Keep it small. A reviewer should be able to read the whole thing in ten minutes.

---

## Framework notes (LangChain vs. LlamaIndex)

> _To be written from the build, not from the docs._ Cover, concretely:
> - What LangChain's abstraction bought (retriever/chain composition, swappable LLM + embedder)
> - Where it added indirection you didn't need
> - How the same retrieve-then-generate would look in LlamaIndex (`VectorStoreIndex.from_documents(...).as_query_engine()`), and which is the better fit for a pipeline this size
> - The honest takeaway: when a framework earns its place vs. when the SDK-direct approach wins

---

## License

MIT © 2026 Nick Lamb
