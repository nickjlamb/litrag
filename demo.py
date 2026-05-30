"""End-to-end demo: ingest -> index -> ask -> answer-with-citations -> grade.

Run after implementing the modules:
    python demo.py

TODO (build session):
- load corpus, build (or load) the FAISS index
- for each example question: rag.answer(...) then faithfulness.grade_answer(...)
- print the answer, the cited sources, and the per-claim grounded verdict
"""

from __future__ import annotations

EXAMPLE_QUESTIONS = [
    # TODO: 2-3 questions answerable from data/sample_abstracts.*,
    # including at least one whose honest answer is "the sources don't say"
    # to show the eval catching an ungrounded claim.
]


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
