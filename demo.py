"""End-to-end demo: ingest -> index -> ask -> answer-with-citations -> grade.

Run after installing requirements and setting ANTHROPIC_API_KEY:
    python demo.py

The corpus load, embedding, and retrieval run locally; only the answer generation and
the faithfulness judge call the LLM API.
"""

from __future__ import annotations

from dotenv import load_dotenv

import faithfulness
import rag
from index import get_or_build_index
from ingest import load_documents

EXAMPLE_QUESTIONS = [
    # Answerable from the SELECT abstract (PMID 37952131 / 38740993).
    (
        "By how much does semaglutide reduce major adverse cardiovascular events in adults "
        "with obesity but without diabetes?"
    ),
    # Answerable from STEP 1 (PMID 33567185).
    (
        "How much weight do adults with overweight or obesity lose on once-weekly "
        "semaglutide 2.4 mg, and what are the most common adverse events?"
    ),
    # Deliberately NOT answerable from the corpus — no abstract covers dementia.
    # The eval should catch any claim the model invents here.
    "Does semaglutide reduce the risk of dementia or cognitive decline?",
]


def _print_grade(grade: dict) -> None:
    verdict = "GROUNDED" if grade["grounded"] else "FLAGGED"
    print(f"  Faithfulness: {verdict}  ({grade['n_claims']} claims, {grade['n_flagged']} flagged)")
    for v in grade["verdicts"]:
        mark = "OK " if v["faithful"] else "XX "
        print(f"    [{mark}] {v['label']:<16} [PMID {v['source']}] {v['claim']}")
        if not v["faithful"]:
            print(f"          quote: \"{v['cited_quote']}\"")
            print(f"          why:   {v['rationale']}")


def main() -> None:
    load_dotenv()

    print("Loading corpus and building/loading FAISS index ...")
    store = get_or_build_index(load_documents())

    for i, question in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"\n{'=' * 80}\nQ{i}: {question}\n{'=' * 80}")
        result = rag.answer(question, store, k=4)
        print(f"\nANSWER: {result['answer']}\n")

        print("  Retrieved:", ", ".join(d.metadata["pmid"] for d in result["retrieved"]))
        grade = faithfulness.grade_answer(result)
        _print_grade(grade)


if __name__ == "__main__":
    main()
