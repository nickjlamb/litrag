"""Build / load the FAISS index from Hugging Face sentence-transformers embeddings.

Embedding and retrieval run fully local — no managed vector-DB key.

TODO (build session):
- get_embedder(model="sentence-transformers/all-MiniLM-L6-v2")
    -> langchain_huggingface.HuggingFaceEmbeddings
- build_index(docs, embedder) -> langchain_community.vectorstores.FAISS
- save_index(store, path="faiss_index/") / load_index(path, embedder)
"""

from __future__ import annotations

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index/"


def get_embedder(model: str = DEFAULT_MODEL):
    """Return a LangChain HuggingFace embeddings wrapper around sentence-transformers."""
    raise NotImplementedError


def build_index(docs, embedder):
    """Embed docs and build a FAISS vector store."""
    raise NotImplementedError


def load_index(path: str = INDEX_PATH, embedder=None):
    """Load a persisted FAISS index from disk."""
    raise NotImplementedError
