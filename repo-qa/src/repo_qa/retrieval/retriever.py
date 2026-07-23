"""
Retrieval.

Responsibility: given a question and a built vector store, return the
top-k most relevant chunks. Kept separate from generation.py so retrieval
quality can be tested/tuned in isolation (which matters — retrieval bugs
are the most common source of bad RAG answers, and you want to be able to
inspect retrieved chunks without also paying for an LLM call every time).
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from repo_qa.config import settings


def retrieve(vectorstore: Chroma, question: str, k: int = None) -> list[tuple[Document, float]]:
    """
    Return (chunk, distance) pairs for the top-k most relevant chunks.

    Note: this is a raw distance score (lower = more similar), not a
    normalized 0-1 relevance score - see docs/concepts.md for why that
    distinction matters when the embedding isn't guaranteed cosine-normalized.
    """
    k = k or settings.default_top_k
    return vectorstore.similarity_search_with_score(question, k=k)
