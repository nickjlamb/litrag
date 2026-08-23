"""LangChain retrieval + generation chain.

Answers a question from retrieved passages AND returns the cited quote per claim,
so the faithfulness eval has something concrete to verify. The structured return is
what separates this from a demo that just emits prose.

The generation step is the only part of the pipeline that calls an LLM API
(``ANTHROPIC_API_KEY``); retrieval is fully local (see ``index.py``).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

GEN_MODEL = "claude-sonnet-4-6"

SYSTEM = """You answer questions strictly from the provided PubMed abstract passages.

Rules:
- Use ONLY the passages. Do not add facts from your own knowledge.
- Break your answer into atomic claims. For each claim, quote — VERBATIM — the exact
  span from a passage that supports it, and give that passage's PMID as the source.
- The cited quote must be copied character-for-character from a passage. Do not
  paraphrase inside the quote.
- If the passages do not answer the question, say so plainly in `answer` and return an
  empty `claims` list rather than inventing support. Do NOT fabricate a quote.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        (
            "human",
            (
                "PASSAGES:\n{context}\n\n"
                "QUESTION: {question}\n\n"
                "Answer the question and return structured claims with verbatim cited quotes."
            ),
        ),
    ]
)


class Claim(BaseModel):
    """One atomic claim in the answer, with the passage span that supports it."""

    text: str = Field(description="The claim, in your own words.")
    cited_quote: str = Field(description="Verbatim span from a passage supporting the claim.")
    source: str = Field(description="PMID of the passage the quote came from.")


class StructuredAnswer(BaseModel):
    """The model's full structured response."""

    answer: str = Field(description="Prose answer to the question, grounded in the passages.")
    claims: list[Claim] = Field(default_factory=list, description="Atomic claims with citations.")


def get_llm():
    """Return the chat model used for generation."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=GEN_MODEL, temperature=0, max_tokens=1024)


def format_context(docs: list[Document]) -> str:
    """Render retrieved passages into a numbered, PMID-labelled context block."""
    blocks = []
    for d in docs:
        pmid = d.metadata.get("pmid", "?")
        title = d.metadata.get("title", "")
        blocks.append(f"[PMID: {pmid}] {title}\n{d.page_content}")
    return "\n\n".join(blocks)


def build_chain(retriever, llm=None) -> Runnable:
    """Compose the retrieval + generation chain.

    A small LangChain LCEL pipeline: ``question -> {context, question} -> prompt ->
    structured LLM``. Returns a runnable that maps a question string to a
    :class:`StructuredAnswer`.
    """
    llm = llm or get_llm()
    structured = llm.with_structured_output(StructuredAnswer)
    return (
        {
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | structured
    )


def answer(question: str, store, k: int = 4) -> dict:
    """Retrieve top-k passages and generate an answer with per-claim citations.

    Returns a plain dict (so downstream eval code stays framework-agnostic):
        {"answer": str,
         "claims": [{"text", "cited_quote", "source"}],
         "retrieved": [Document, ...]}
    """
    retriever = store.as_retriever(search_kwargs={"k": k})
    retrieved = retriever.invoke(question)
    chain = build_chain(retriever)
    result: StructuredAnswer = chain.invoke(question)
    return {
        "answer": result.answer,
        "claims": [c.model_dump() for c in result.claims],
        "retrieved": retrieved,
    }


if __name__ == "__main__":
    from index import get_or_build_index
    from ingest import load_documents

    store = get_or_build_index(load_documents())
    out = answer("By how much does semaglutide reduce major adverse cardiovascular events?", store)
    print("ANSWER:", out["answer"], "\n")
    for c in out["claims"]:
        print(f"- {c['text']}\n    quote: \"{c['cited_quote']}\"  [PMID {c['source']}]")
