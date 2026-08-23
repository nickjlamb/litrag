"""Watch Stage 1 catch a fabricated citation — no API key, no judge call.

The eval's first stage is a deterministic quote locator: a claim's cited quote
must actually exist (verbatim, modulo whitespace/case) in the retrieved source.
A fabricated quote fails the locator and is flagged as `hallucinated_quote`
before any LLM judge is invoked — hallucination detection for free.

    python examples/03_catch_a_hallucination.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faithfulness import locate  # noqa: E402

SOURCE = (
    "A primary cardiovascular end-point event occurred in 569 of the 8803 patients "
    "(6.5%) in the semaglutide group and in 701 of the 8801 patients (8.0%) in the "
    "placebo group (hazard ratio, 0.80; 95% confidence interval, 0.72 to 0.90; P<0.001)."
)

CASES = [
    ("verbatim quote", "hazard ratio, 0.80; 95% confidence interval, 0.72 to 0.90"),
    ("fuzzy match (case/spacing drift)", "Hazard Ratio, 0.80;  95% confidence  interval, 0.72 to 0.90"),
    ("paraphrase (rejected — not verbatim)", "the drug cut heart attacks by about a fifth versus placebo"),
    ("fabricated quote", "semaglutide increased all-cause mortality by 35%"),
]


def main() -> None:
    print("Source passage (from the SELECT trial abstract):\n")
    print(f"  {SOURCE}\n")
    for name, quote in CASES:
        hit = locate(quote, SOURCE)
        status = "LOCATED " if hit["found"] else "FLAGGED "
        print(f"[{status}] {name}  (score={hit['score']:.1f})")
        print(f"           \"{quote}\"\n")

    print(
        "Only located quotes proceed to Stage 2 (the LLM-as-judge). Everything else\n"
        "is flagged as `hallucinated_quote` without spending a single API call."
    )


if __name__ == "__main__":
    main()
