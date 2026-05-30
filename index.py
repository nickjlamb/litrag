"""Build / load the FAISS index from Hugging Face sentence-transformers embeddings.

Embedding and retrieval run fully local — no managed vector-DB key. The first call
downloads the sentence-transformers model (~90 MB) from Hugging Face; after that it
runs offline.
"""

from __future__ import annotations

import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index/"


def get_embedder(model: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    """Return a LangChain HuggingFace embeddings wrapper around sentence-transformers.

    Normalized embeddings so FAISS inner-product ≈ cosine similarity.
    """
    return HuggingFaceEmbeddings(
        model_name=model,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(docs: list[Document], embedder: HuggingFaceEmbeddings | None = None) -> FAISS:
    """Embed ``docs`` and build an in-memory FAISS vector store."""
    embedder = embedder or get_embedder()
    return FAISS.from_documents(docs, embedder)


def save_index(store: FAISS, path: str = INDEX_PATH) -> None:
    """Persist a FAISS index to disk."""
    store.save_local(path)


def load_index(path: str = INDEX_PATH, embedder: HuggingFaceEmbeddings | None = None) -> FAISS:
    """Load a persisted FAISS index from disk.

    ``allow_dangerous_deserialization`` is required because FAISS stores the docstore
    as a pickle; it is safe here since we only ever load an index this repo wrote.
    """
    embedder = embedder or get_embedder()
    return FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)


def get_or_build_index(
    docs: list[Document],
    path: str = INDEX_PATH,
    embedder: HuggingFaceEmbeddings | None = None,
    rebuild: bool = False,
) -> FAISS:
    """Load the index from ``path`` if present, otherwise build it from ``docs`` and save.

    The common entry point for the demo: embed once, reuse thereafter.
    """
    embedder = embedder or get_embedder()
    if not rebuild and os.path.isdir(path):
        return load_index(path, embedder)
    store = build_index(docs, embedder)
    save_index(store, path)
    return store


if __name__ == "__main__":
    from ingest import load_documents

    docs = load_documents()
    print(f"Embedding {len(docs)} passages with {DEFAULT_MODEL} ...")
    store = get_or_build_index(docs, rebuild=True)
    print(f"Built FAISS index, saved to {INDEX_PATH}")

    hits = store.similarity_search("Does semaglutide reduce cardiovascular events?", k=3)
    print("\nTop hits for a sample query:")
    for d in hits:
        print(f"  [{d.metadata['pmid']}] {d.metadata['title'][:70]}")
