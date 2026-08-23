# Contributing to LitRAG

Thanks for your interest in improving LitRAG. This is a small, readable codebase on purpose — a reviewer should be able to read the whole pipeline in about ten minutes — and contributions that keep it that way are the most welcome kind.

## Dev setup

```bash
git clone https://github.com/nickjlamb/litrag.git && cd litrag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ruff
```

Python 3.10+ is required. An `ANTHROPIC_API_KEY` (in `.env`, copied from `.env.example`) is only needed for generation and the faithfulness judge — ingestion, embedding, retrieval, and the test suite all run without one.

## Running tests

```bash
pytest
```

The test suite covers the deterministic quote locator (`faithfulness.locate` / `locate_quote`) and runs entirely offline — no API key, no model download. If you add behavior, add a test; if you fix a bug, add the test that would have caught it.

## Linting

```bash
ruff check .
```

CI runs both `pytest` and `ruff check` on every push and pull request; both must pass.

## What makes a good contribution

**Good fits:** items on the [roadmap](README.md#roadmap); bug fixes with a regression test; corpus additions in the documented `data/` format (`## PMID:` / `**Title:**` / `**Source:**` blocks, real abstracts only); improvements to the eval's precision (locator thresholds, judge rubric) backed by examples; docs fixes.

**Talk first (open an issue):** new dependencies, new pipeline stages, framework swaps, or anything that grows the core beyond its five files. The bar is whether the change earns its complexity.

**Out of scope:** features that make the pipeline bigger without making the groundedness story better. LitRAG is not trying to become a general-purpose RAG framework.

## Pull request guidelines

1. Fork, branch from `main`, keep the branch focused on one change.
2. Make sure `pytest` and `ruff check .` pass locally.
3. Update `CHANGELOG.md` under **Unreleased** with a one-line description of the change.
4. Write a PR description that says *why*, not just *what*. Small PRs get reviewed fast; large ones get questions.

## Reporting issues

Use the issue templates. For bugs, include the question you asked, the corpus you used, and the full verdict output — the eval's verdicts (`label`, `locate_score`, `rationale`) are usually enough to localize a problem quickly.

## Code style

Follow the patterns already in the codebase: type hints, docstrings that explain *why*, `from __future__ import annotations`, and plain dicts at module boundaries so downstream code stays framework-agnostic. Ruff (default rules) is the arbiter of formatting disputes.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
