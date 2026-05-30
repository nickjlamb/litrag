"""Citation-faithfulness eval — the groundedness layer that makes this repo worth more
than a stock RAG demo.

For each claim the model makes:
  1. Locate `cited_quote` in the retrieved source (deterministic: normalize +
     rapidfuzz.partial_ratio). If it can't be located, the quote is fabricated ->
     short-circuit to "hallucinated_quote" without spending a judge call.
  2. If located, use an LLM-as-judge to grade whether the source SUPPORTS the claim:
     supports / partial / contradicts / not_found.

Port the locator + judge logic from the cookbook citation_faithfulness.py
(/Users/NickLamb/jobs/cookbook-contribution/citation_faithfulness.py).

TODO (build session):
- locate_quote(quote, source, threshold=90.0) -> bool
- grade_support(claim, passage) -> {"support", "rationale", "confidence"}  (forced tool use)
- grade_answer(rag_result) -> per-claim verdicts + an overall grounded? flag
"""

from __future__ import annotations

JUDGE_MODEL = "claude-sonnet-4-6"
FUZZY_THRESHOLD = 90.0


def locate_quote(quote: str, source: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """True if `quote` can be located in `source` (exact or fuzzy)."""
    raise NotImplementedError


def grade_support(claim: str, passage: str) -> dict:
    """LLM-as-judge: does `passage` support `claim`? Returns support level + rationale."""
    raise NotImplementedError


def grade_answer(rag_result: dict) -> dict:
    """Grade every claim in a rag.answer() result; return verdicts + overall grounded flag."""
    raise NotImplementedError
