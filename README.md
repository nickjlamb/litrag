<div align="center">

# LitRAG

**Retrieval-augmented generation over the medical literature — with a citation-faithfulness eval built in.**

[![CI](https://github.com/nickjlamb/litrag/actions/workflows/ci.yml/badge.svg)](https://github.com/nickjlamb/litrag/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Website](https://img.shields.io/badge/site-pharmatools.ai%2Flitrag-6f42c1)](https://www.pharmatools.ai/litrag)

</div>

---

Most RAG demos stop at "it retrieved something and wrote an answer." LitRAG goes one step further: **it checks whether the generated answer is actually supported by the retrieved sources**, and flags hallucinated or unsupported claims. That groundedness layer — not the pipeline — is the point.

The model must return, for every claim it makes, a **verbatim quote** from a retrieved passage plus the PMID it came from. A two-stage eval then verifies each citation:

1. **Locate** — a deterministic string check (normalize + fuzzy match) confirms the quote really exists in the source. A fabricated quote is caught here, for free, before any judge is called.
2. **Judge** — an LLM-as-judge grades whether the located passage actually *supports* the claim: `supports` / `partial` / `contradicts` / `not_found`.

Embedding and retrieval run fully local (Hugging Face `sentence-transformers` + FAISS — no managed vector-DB key). Only generation and the judge call an LLM API.

## Architecture

```mermaid
flowchart TB
    subgraph local["Local — no API key required"]
        A[PubMed abstracts<br/><code>data/</code>] --> B[Chunk + attach<br/><code>pmid / title / source</code>]
        B --> C[sentence-transformers<br/>embeddings]
        C --> D[(FAISS index)]
        Q([Question]) --> R[Retrieve top-k]
        D --> R
    end

    subgraph api["LLM API"]
        R --> G["Generate structured answer<br/><code>{answer, claims:[{text, cited_quote, source}]}</code>"]
        G --> L{"Stage 1 — Locate quote<br/>in retrieved source<br/><i>(deterministic, fuzzy match)</i>"}
        L -- "not found" --> H[/"hallucinated_quote<br/>flagged, no judge call"/]
        L -- found --> J["Stage 2 — LLM-as-judge<br/>supports / partial /<br/>contradicts / not_found"]
        J --> V[/"per-claim verdict +<br/>overall grounded flag"/]
    end

    style H fill:#fdd,stroke:#c33
    style V fill:#dfd,stroke:#3a3
```

Built on [LangChain](https://python.langchain.com/) (orchestration), [Hugging Face sentence-transformers](https://www.sbert.net/) (embeddings), and [FAISS](https://github.com/facebookresearch/faiss) (vector store). The judge deliberately drops to the raw [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) for forced tool use and prompt caching — see [Framework notes](docs/framework-notes.md) for why.

## Quick start

```bash
git clone https://github.com/nickjlamb/litrag.git && cd litrag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # add ANTHROPIC_API_KEY for generation + judge
python demo.py
```

The demo ingests a corpus of 15 real semaglutide abstracts, builds the FAISS index (first run downloads the ~90 MB embedding model, then it's offline), asks three questions — two answerable from the corpus, one deliberately *not* — and grades every claim (output abridged):

```text
================================================================================
Q1: By how much does semaglutide reduce major adverse cardiovascular events ...
================================================================================

ANSWER: In adults with overweight or obesity and established cardiovascular
disease but without diabetes, semaglutide reduced major adverse cardiovascular
events by 20% versus placebo (hazard ratio 0.80; 95% CI 0.72–0.90). ...

  Retrieved: 37952131, 38740993, 33567185, 34706925
  Faithfulness: GROUNDED  (3 claims, 0 flagged)
    [OK ] supported        [PMID 37952131] Semaglutide reduced MACE by 20% vs placebo
    ...

================================================================================
Q3: Does semaglutide reduce the risk of dementia or cognitive decline?
================================================================================

ANSWER: The provided passages do not address dementia or cognitive decline. ...
  Faithfulness: GROUNDED  (0 claims, 0 flagged)   # honest abstention — no invented support
```

**No API key?** Retrieval works without one:

```bash
python index.py    # builds the index and runs a sample similarity search, fully local
```

## Examples

Runnable, focused scripts live in [`examples/`](examples/):

| Example | What it shows | API key |
|---|---|---|
| [`01_retrieval_only.py`](examples/01_retrieval_only.py) | Build/load the index and inspect top-k passages for a query | none |
| [`02_ask_with_citations.py`](examples/02_ask_with_citations.py) | One question end-to-end: structured answer with per-claim verbatim quotes | required |
| [`03_catch_a_hallucination.py`](examples/03_catch_a_hallucination.py) | Feed the eval a fabricated citation and watch Stage 1 flag it — no judge call needed | none |
| [`04_bring_your_own_corpus.py`](examples/04_bring_your_own_corpus.py) | Point the pipeline at your own abstracts file | none for retrieval |

## Project layout

| File | Role |
|------|------|
| [`ingest.py`](ingest.py) | Load the abstract corpus, chunk into passages with `{pmid, title, source}` metadata |
| [`index.py`](index.py) | Build/load the FAISS index from `sentence-transformers` embeddings |
| [`rag.py`](rag.py) | LangChain retrieval + generation chain; returns the answer **with** per-claim cited quotes |
| [`faithfulness.py`](faithfulness.py) | Two-stage citation-faithfulness eval: deterministic quote locator → LLM-as-judge |
| [`demo.py`](demo.py) | End-to-end run: ingest → index → ask → answer → grade |
| [`data/`](data/) | 15 real semaglutide abstracts (STEP, SELECT, SUSTAIN…) so the repo runs key-free for retrieval |

A reviewer can read the whole pipeline in about ten minutes — that's deliberate.

## Why the eval is two-stage

A single LLM-as-judge can be argued with; a string match can't. Requiring a verbatim quote per claim turns hallucination detection into two cheap, complementary checks:

- **The locator is unspoofable.** If the model invents a quote, `rapidfuzz.partial_ratio` against the retrieved passages fails, and the claim is flagged as `hallucinated_quote` without spending a judge call. Paraphrases below the 90-point threshold are rejected too — verbatim means verbatim.
- **The judge grades only from the passage.** With the quote located, the judge answers a narrower, easier question: does *this passage* support *this claim*? It's forbidden from using outside knowledge, forced into a structured verdict via tool use, and the passage block is prompt-cached so grading many claims against one source is cheap.

An answer that honestly abstains ("the sources don't say") makes no claims and is grounded by definition.

## Framework notes

The pipeline is implemented in LangChain; [docs/framework-notes.md](docs/framework-notes.md) is an honest, from-the-build write-up of where the framework earned its keep (`Document` metadata plumbing, `with_structured_output`) and where the project dropped to the raw SDK on purpose (the judge needs forced tool use + per-block prompt caching that the abstraction hides). It also maps the design onto LlamaIndex primitive-by-primitive and explains why swapping frameworks isn't a free lunch here.

## Roadmap

- [ ] **Live corpus ingestion** — pull fresh abstracts from PubMed via NCBI E-utilities / the [PubCrawl](https://www.pharmatools.ai/pubcrawl) MCP server instead of the static sample
- [ ] **LlamaIndex variant** — a small `rag_llamaindex.py` so the framework comparison is code, not prose
- [ ] **Benchmark the eval** — score the faithfulness judge against RAGAS and DeepEval faithfulness metrics on a labelled claim set
- [ ] **Larger corpora** — beyond one drug: multi-topic corpora and retrieval quality metrics (recall@k against known-relevant PMIDs)
- [ ] **CLI** — `litrag ask "..."` entry point with `--corpus` and `--judge` flags

Suggestions welcome — open an [issue](https://github.com/nickjlamb/litrag/issues).

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, test and lint commands, and PR guidelines. Notable changes are tracked in [CHANGELOG.md](CHANGELOG.md).

```bash
pip install -r requirements.txt && pip install ruff
pytest            # quote-locator tests run without any API key
ruff check .
```

## Citation

If you use LitRAG in your work, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{lamb_litrag_2026,
  author = {Lamb, Nick},
  title  = {LitRAG: grounded literature RAG with a citation-faithfulness eval},
  year   = {2026},
  url    = {https://github.com/nickjlamb/litrag}
}
```

## License

[MIT](LICENSE) © 2026 [Nick Lamb](https://www.pharmatools.ai)
