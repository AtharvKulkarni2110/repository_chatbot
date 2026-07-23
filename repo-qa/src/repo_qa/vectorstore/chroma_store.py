"""
Vector store management.

Responsibility: wrap Chroma so the rest of the codebase never talks to
Chroma's API directly. If we ever swap Chroma for Qdrant/Pinecone/pgvector,
this is the only file that should need to change.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from repo_qa.config import settings


def build_index(chunks: list[Document], embedding_fn: Embeddings, persist_dir: str) -> Chroma:
    """Embed and persist a fresh Chroma collection from the given chunks."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=persist_dir,
        collection_name=settings.collection_name,
    )


def load_index(embedding_fn: Embeddings, persist_dir: str) -> Chroma:
    """Load an already-built Chroma collection from disk."""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_fn,
        collection_name=settings.collection_name,
    )
