"""One question, end-to-end: retrieve, answer with per-claim citations, grade.

Requires ANTHROPIC_API_KEY in your environment or .env — generation and the
faithfulness judge are the only steps that call an LLM API.

    python examples/02_ask_with_citations.py "How much weight do adults lose on semaglutide?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

import faithfulness
import rag
from index import get_or_build_index
from ingest import load_documents

DEFAULT_QUESTION = (
    "How much weight do adults with overweight or obesity lose on once-weekly "
    "semaglutide 2.4 mg?"
)


def main() -> None:
    load_dotenv()
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    store = get_or_build_index(load_documents())
    result = rag.answer(question, store, k=4)

    print(f"Q: {question}\n")
    print(f"ANSWER: {result['answer']}\n")
    print("Claims and their verbatim citations:")
    for c in result["claims"]:
        print(f"- {c['text']}")
        print(f"    quote:  \"{c['cited_quote']}\"")
        print(f"    source: PMID {c['source']}")

    grade = faithfulness.grade_answer(result)
    verdict = "GROUNDED" if grade["grounded"] else "FLAGGED"
    print(f"\nFaithfulness: {verdict} ({grade['n_claims']} claims, {grade['n_flagged']} flagged)")


if __name__ == "__main__":
    main()
