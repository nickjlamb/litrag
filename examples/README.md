# Examples

Focused, runnable scripts — each one demonstrates a single slice of the pipeline. Run them from the repo root (they add it to `sys.path` themselves):

```bash
python examples/01_retrieval_only.py        # local retrieval, no API key
python examples/03_catch_a_hallucination.py # the Stage-1 locator, no API key
python examples/04_bring_your_own_corpus.py # custom corpus format, no API key
python examples/02_ask_with_citations.py    # full pipeline — needs ANTHROPIC_API_KEY
```

| Example | What it shows | API key |
|---|---|---|
| `01_retrieval_only.py` | Build/load the FAISS index; inspect top-k passages with scores and provenance | none |
| `02_ask_with_citations.py` | One question end-to-end: structured answer, per-claim verbatim quotes, faithfulness grade | required |
| `03_catch_a_hallucination.py` | The deterministic quote locator flagging fabricated and paraphrased citations | none |
| `04_bring_your_own_corpus.py` | The corpus file format, ingestion, and an index over your own abstracts | none |
