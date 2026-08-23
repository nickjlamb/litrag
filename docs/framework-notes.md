# Framework notes: LangChain vs. LlamaIndex

_Written from the build, not the docs._

LitRAG's pipeline is implemented in LangChain. This document records where the framework earned its keep and where it added indirection — and then maps the same design onto LlamaIndex, primitive by primitive, with the trade-offs that mapping would surface. The goal is an honest comparison grounded in a real build, not a feature-matrix shootout.

## What LangChain bought

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

## Where it added indirection

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

## LitRAG in LlamaIndex — a design read

_I built the pipeline in LangChain; this section is how I'd map it onto LlamaIndex and
the trade-offs I'd expect — **not** a second, shipped implementation. Kept deliberately
separate from the build notes above, which are from the build._

Retrieve-then-generate collapses to roughly:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
index = VectorStoreIndex.from_documents(SimpleDirectoryReader("data/").load_data())
print(index.as_query_engine().query("How much weight loss with semaglutide?"))
```

Mapping LitRAG's pieces onto LlamaIndex primitives:

| LitRAG (LangChain) | LlamaIndex equivalent |
|--------------------|-----------------------|
| `Document(page_content, metadata={pmid, title, source})` | `TextNode` + `metadata`, with `excluded_embed_metadata_keys` to keep PMIDs out of the embedded text |
| `ingest.chunk` (word-window splitter) | a `SentenceSplitter` / `TokenTextSplitter` node parser |
| `HuggingFaceEmbeddings` | `HuggingFaceEmbedding` (same sentence-transformers model) |
| `FAISS.from_documents` | `VectorStoreIndex` over a `FaissVectorStore` |
| `store.as_retriever(search_kwargs={"k": 4})` | `index.as_retriever(similarity_top_k=4)` |
| `llm.with_structured_output(StructuredAnswer)` | `index.as_query_engine(output_cls=StructuredAnswer)` (Pydantic program) |
| hand-rolled per-claim `cited_quote` contract | `CitationQueryEngine` — citations are first-class; the response carries `source_nodes` |

**What it would buy.** Citation is native. `CitationQueryEngine` numbers its sources and
hands back the `source_nodes` it used, so "answer with provenance" isn't something I bolt
on with a Pydantic schema — it's the default contract. For a product whose entire point
is grounded citations, that's a genuine fit advantage.

**What would stay exactly the same.** The faithfulness judge. `faithfulness.grade_support`
would still drop to the raw Anthropic SDK for *forced* tool use + `cache_control` on the
passage — LlamaIndex abstracts those away just as LangChain does. The honest seam
(framework for retrieval, SDK for the eval that *is* the product) is identical either
way; the framework choice only ever touches the boring 80%.

**What it would cost.** A second embedding/index stack to maintain, and — the real catch —
LlamaIndex's citation synthesizer re-chunks and renumbers sources its own way, which
fights the contract the eval depends on. The locator needs the model to quote a span
*verbatim* from a retrieved passage; a synthesizer that paraphrases into a numbered
citation breaks `locate_quote` before the judge ever runs. Reconciling first-class
citations with a verbatim-quote requirement is non-trivial, and it's the main reason
swapping frameworks isn't a free lunch here.

**Honest status.** Not built. If the LlamaIndex equivalent matters for a given purpose,
the right move is a small `rag_llamaindex.py` variant — code, not more prose. (This is
on the [roadmap](../README.md#roadmap).)

## The honest takeaway

For a 5-file pipeline the framework earns its place on the *glue* — the
`Document`/metadata plumbing and `with_structured_output` — and costs a layer exactly
where the product lives: the faithfulness judge, which I built on the Anthropic SDK
directly for forced tool use + caching. Neither framework touches the two things that
make this repo more than a demo (the per-claim `cited_quote` contract and the two-stage
groundedness check) — those are plain Pydantic + the SDK. So: reach for the framework
for the boring 80% (load → embed → retrieve → structure); drop to the SDK for the 20%
that *is* the point. If retrieval-with-citations were the whole product, LlamaIndex
would be the better-fitting default at this size.
