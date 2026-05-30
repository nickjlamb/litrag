# LitRAG

**Grounded retrieval-augmented generation over the medical literature — with a citation-faithfulness eval built in.**

Most RAG demos stop at "it retrieved something and wrote an answer." LitRAG goes one step further: it *checks whether the generated answer is actually supported by the retrieved sources*, and flags hallucinated or unsupported claims. That groundedness layer — not the pipeline — is the point.

Built on **LangChain** (orchestration) + **Hugging Face** `sentence-transformers` (embeddings) + **FAISS** (vector store), so it runs locally with no managed vector-DB key required.

> Status: built. Corpus = 15 real semaglutide abstracts (see `data/`). Retrieval runs
> locally and is verified; generation + faithfulness judge need an LLM key (see Quickstart).

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

_Written from the build, not the docs._

**What LangChain bought.**

- **`Document` + metadata as the universal currency.** `ingest.py` emits
  `Document(page_content, metadata={pmid, title, source})`; FAISS embeds it, the
  retriever returns it, and the `{pmid, title, source}` rides through embedding and
  retrieval untouched. The faithfulness eval needs exactly that provenance, and the
  framework carried it end-to-end for free — no parallel bookkeeping of "which text
  came from which abstract."
- **Swappable embedder + LLM.** `HuggingFaceEmbeddings` and `ChatAnthropic` are drop-in;
  switching the judge/generator to OpenAI is a one-line import change. The local
  embedder and the API generator sit behind the same interfaces.
- **`with_structured_output(PydanticModel)`.** This is the biggest win for *this*
  pipeline. Structured per-claim citations (`{answer, claims:[{text, cited_quote,
  source}]}`) are the whole point, and I got validated objects back without hand-writing
  a tool schema or a parser — just a Pydantic model.
- **FAISS persistence.** `from_documents` / `save_local` / `load_local` gave embed-once,
  reuse-after for nothing (see `index.get_or_build_index`).

**Where it added indirection.**

- **LCEL composition is clean until you debug it.** `{"context": retriever | format,
  "question": passthrough} | prompt | structured` reads well, but the dict→Runnable
  coercion is opaque: drop a non-`Runnable` into that dict and you get a cryptic
  `Expected a Runnable, callable or dict` from deep in `coerce_to_runnable`, far from
  the line you wrote. (Hit this verbatim while wiring a test stub.)
- **The judge left the framework on purpose.** `faithfulness.grade_support` uses the raw
  Anthropic SDK, not LangChain — because it wants *forced* tool use **and**
  `cache_control` on the source-passage content block (so checking many claims against
  one long abstract reuses the cached passage). `with_structured_output` abstracts the
  tool away, but it also abstracts away per-block cache control — so the one place I most
  wanted provider-specific control is the place I dropped out of the abstraction. That's
  the honest seam.
- **`HuggingFaceEmbeddings` is a thin wrapper.** For a corpus this small you could call
  `sentence-transformers` + `faiss` directly in ~20 lines and lose almost nothing.

**The same thing in LlamaIndex.** Retrieve-then-generate collapses to roughly:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
index = VectorStoreIndex.from_documents(SimpleDirectoryReader("data/").load_data())
print(index.as_query_engine().query("How much weight loss with semaglutide?"))
```

Fewer lines, retrieval and synthesis fused, and LlamaIndex's node model +
`CitationQueryEngine` arguably fit "answer with sources" more natively, since citation is
a first-class concern there rather than something you bolt on with a Pydantic schema.

**The honest takeaway.** For a 5-file pipeline the framework earns its place on the
*glue* — the `Document`/metadata plumbing and `with_structured_output` — and costs a
layer exactly where the product lives: the faithfulness judge, which I built on the
Anthropic SDK directly for forced tool use + caching. Neither framework touches the two
things that make this repo more than a demo (the per-claim `cited_quote` contract and the
two-stage groundedness check) — those are plain Pydantic + the SDK. So: reach for the
framework for the boring 80% (load → embed → retrieve → structure); drop to the SDK for
the 20% that *is* the point. If retrieval-with-citations were the whole product,
LlamaIndex would be the better-fitting default at this size.

---

## License

MIT © 2026 Nick Lamb
