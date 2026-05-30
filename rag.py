"""LangChain retrieval + generation chain.

Answers a question from retrieved passages AND returns the cited quote per claim,
so the faithfulness eval has something concrete to verify. The structured return is
what separates this from a demo that just emits prose.

TODO (build session):
- get_llm(): langchain_anthropic.ChatAnthropic(model="claude-sonnet-4-6")
    (or langchain_openai.ChatOpenAI)
- build_chain(retriever, llm): a LangChain chain prompted to answer and, per claim,
    quote the supporting passage verbatim
- answer(question, store, k=4) -> {
        "answer": str,
        "claims": [{"text": str, "cited_quote": str, "source": str}],
        "retrieved": [Document, ...],
    }
"""

from __future__ import annotations

GEN_MODEL = "claude-sonnet-4-6"


def get_llm():
    """Return the chat model used for generation."""
    raise NotImplementedError


def build_chain(retriever, llm):
    """Compose the retrieval + generation chain."""
    raise NotImplementedError


def answer(question: str, store, k: int = 4) -> dict:
    """Retrieve top-k passages and generate an answer with per-claim citations."""
    raise NotImplementedError
