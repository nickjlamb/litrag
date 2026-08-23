"""Citation-faithfulness eval — the groundedness layer that makes this repo worth more
than a stock RAG demo.

For each claim the model makes:
  1. Locate `cited_quote` in the retrieved source (deterministic: normalize +
     rapidfuzz.partial_ratio). If it can't be located, the quote is fabricated ->
     short-circuit to "hallucinated_quote" without spending a judge call.
  2. If located, use an LLM-as-judge to grade whether the source SUPPORTS the claim:
     supports / partial / contradicts / not_found.

Locator + judge logic ported from a cookbook citation-faithfulness notebook: a cheap,
unspoofable string check in front of an LLM judge that grades ONLY from the passage.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

JUDGE_MODEL = "claude-sonnet-4-6"
FUZZY_THRESHOLD = 90.0


def _norm(s: str) -> str:
    """Normalize whitespace and case so trivial formatting differences don't fail a match."""
    return re.sub(r"\s+", " ", s or "").strip().casefold()


def locate(quote: str, source: str, threshold: float = FUZZY_THRESHOLD) -> dict:
    """Locate ``quote`` in ``source``; return {"found", "score"}.

    Exact (normalized) substring first, then a fuzzy ``partial_ratio`` fallback tolerant
    of OCR/whitespace drift. The threshold stops paraphrases from passing as verbatim.
    """
    nq, ns = _norm(quote), _norm(source)
    if not nq:
        return {"found": False, "score": 0.0}
    if nq in ns:
        return {"found": True, "score": 100.0}
    score = float(fuzz.partial_ratio(nq, ns))
    return {"found": score >= threshold, "score": score}


def locate_quote(quote: str, source: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """True if ``quote`` can be located in ``source`` (exact or fuzzy)."""
    return locate(quote, source, threshold)["found"]


# --- Stage 2: LLM-as-judge support grading (forced tool use for structured output) ---

RUBRIC = """You are a strict grader assessing whether a SOURCE PASSAGE supports a CLAIM.

Judge ONLY using the SOURCE PASSAGE. Do NOT use any outside knowledge, even if you believe the
claim is true or false in general. If the passage does not contain the information, that is
`not_found`, regardless of what you know.

Grade the support level:
- supports:    the passage directly establishes the claim.
- partial:     the passage is consistent with the claim but does not fully establish it
               (e.g. a surrogate endpoint, non-inferiority used to imply superiority, symptomatic
               relief used to imply cure).
- contradicts: the passage states the opposite of the claim.
- not_found:   the passage is about something else and neither supports nor contradicts the claim.

In your rationale, quote the specific phrase from the passage that decides your verdict."""

VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit the support verdict for the claim given the source passage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "support": {
                "type": "string",
                "enum": ["supports", "partial", "contradicts", "not_found"],
            },
            "rationale": {
                "type": "string",
                "description": "1-2 sentences; quote the deciding phrase from the passage.",
            },
            "confidence": {
                "type": "number",
                "description": "0.0-1.0 confidence in the verdict.",
            },
        },
        "required": ["support", "rationale", "confidence"],
    },
}


def grade_support(claim: str, passage: str, temperature: float = 0.0) -> dict:
    """LLM-as-judge: does ``passage`` support ``claim``? Returns support level + rationale.

    Uses forced tool use for validated structured output, and caches the (potentially
    long) passage so checking several claims against one source reuses it.
    """
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        temperature=temperature,
        system=RUBRIC,
        tools=[VERDICT_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdict"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"SOURCE PASSAGE:\n{passage}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"CLAIM:\n{claim}\n\nCall submit_verdict with your graded support level.",
                    },
                ],
            }
        ],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Judge did not return a verdict tool call")


# --- Wire stages 1+2 over a rag.answer() result ---

# A citation is faithful only if the quote is located AND support == "supports"
# (optionally "partial"). These map onto the cookbook's four gold categories.
_SUPPORT_TO_LABEL = {
    "supports": "supported",
    "contradicts": "contradicted",
    "partial": "unsupported",
    "not_found": "unsupported",
}


def _passage_for(source: str, retrieved, fallback_all: bool = True) -> str:
    """Collect the retrieved passage text for a claim's cited PMID.

    If the cited PMID isn't among the retrieved docs, fall back to the full retrieved
    set — so a quote cited to the "wrong" source is still located if it exists at all,
    and only a genuinely fabricated quote short-circuits to hallucinated.
    """
    matches = [d.page_content for d in retrieved if str(d.metadata.get("pmid")) == str(source)]
    if matches:
        return "\n\n".join(matches)
    if fallback_all:
        return "\n\n".join(d.page_content for d in retrieved)
    return ""


def grade_claim(claim: dict, retrieved, allow_partial: bool = False) -> dict:
    """Grade a single claim: locate its quote, then (if located) judge support."""
    passage = _passage_for(claim["source"], retrieved)
    loc = locate(claim["cited_quote"], passage)
    if not loc["found"]:
        return {
            "claim": claim["text"],
            "cited_quote": claim["cited_quote"],
            "source": claim["source"],
            "label": "hallucinated_quote",
            "faithful": False,
            "located": False,
            "locate_score": round(loc["score"], 1),
            "support": "not_found",
            "rationale": "Cited quote not found in the retrieved source.",
            "confidence": 1.0,
        }
    v = grade_support(claim["text"], passage)
    faithful = v["support"] == "supports" or (allow_partial and v["support"] == "partial")
    return {
        "claim": claim["text"],
        "cited_quote": claim["cited_quote"],
        "source": claim["source"],
        "label": _SUPPORT_TO_LABEL[v["support"]],
        "faithful": faithful,
        "located": True,
        "locate_score": round(loc["score"], 1),
        **v,
    }


def grade_answer(rag_result: dict, allow_partial: bool = False) -> dict:
    """Grade every claim in a rag.answer() result; return verdicts + overall grounded flag.

    ``grounded`` is True when every claim is faithful. An answer that honestly abstains
    (empty claims, e.g. "the sources don't say") is grounded by definition — it made no
    unsupported claim.
    """
    retrieved = rag_result.get("retrieved", [])
    verdicts = [grade_claim(c, retrieved, allow_partial) for c in rag_result.get("claims", [])]
    flagged = [v for v in verdicts if not v["faithful"]]
    return {
        "grounded": len(flagged) == 0,
        "n_claims": len(verdicts),
        "n_flagged": len(flagged),
        "verdicts": verdicts,
    }


if __name__ == "__main__":
    # Stage-1 smoke test (no API key needed).
    src = "Semaglutide reduced major adverse cardiovascular events by 20%."
    print("exact :", locate_quote("reduced major adverse cardiovascular events by 20%", src))
    print("fuzzy :", locate_quote("Reduced  major adverse cardiovascular events by 20%", src))
    print("fake  :", locate_quote("increased all-cause mortality by 35%", src))
