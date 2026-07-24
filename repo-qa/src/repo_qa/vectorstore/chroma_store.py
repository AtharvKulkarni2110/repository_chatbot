"""
Vector store management.

Responsibility: wrap Chroma so the rest of the codebase never talks to
Chroma's API directly. If we ever swap Chroma for Qdrant/Pinecone/pgvector,
this is the only file that should need to change.

Embedding mismatch guard
-------------------------
Chroma will hard-crash if you try to add a vector of a different dimension
than what a collection already holds (verified: "Collection expecting
embedding with dimension of 3, got 5"). That protects against silent
corruption, but only once vectors are already in the collection, and the
error message assumes you already know why dimensions differ.

The real risk: this project can embed with two different, incompatible
schemes (a real HuggingFace model vs. the offline TF-IDF fallback), and
which one gets used can silently change between runs depending on internet
availability. Mixing them in the same collection is meaningless - the
vectors were never comparable in the first place, dimension crash or not.

The fix: at build time, stamp a small manifest file (embedding_manifest.json)
into persist_dir naming exactly which embedding scheme produced the index.
At both build and load time, check any existing manifest against the
embedding scheme about to be used, and refuse with a clear, actionable
error on mismatch - before Chroma ever gets a chance to crash confusingly.
"""

import json
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from repo_qa.config import settings
from repo_qa.embeddings.provider import get_embedding_identifier

MANIFEST_FILENAME = "embedding_manifest.json"


class EmbeddingMismatchError(RuntimeError):
    """Raised when the embedding scheme in use doesn't match what an existing index was built with."""


def _manifest_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, MANIFEST_FILENAME)


def _read_manifest(persist_dir: str) -> dict | None:
    path = _manifest_path(persist_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _write_manifest(persist_dir: str, identifier: str, dimension: int) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    with open(_manifest_path(persist_dir), "w") as f:
        json.dump({"embedding_identifier": identifier, "dimension": dimension}, f, indent=2)


def _check_for_mismatch(persist_dir: str, embedding_fn: Embeddings) -> str:
    """Compare the embedding scheme about to be used against any existing manifest. Returns the current identifier."""
    identifier = get_embedding_identifier(embedding_fn)
    existing = _read_manifest(persist_dir)

    if existing is not None and existing["embedding_identifier"] != identifier:
        raise EmbeddingMismatchError(
            f"Index at '{persist_dir}' was built with embedding scheme "
            f"'{existing['embedding_identifier']}', but this run would use "
            f"'{identifier}'. These produce incompatible vectors and can't be "
            f"mixed in one index (this is often caused by internet being "
            f"available on one run and not another, silently switching between "
            f"the real model and the offline TF-IDF fallback).\n"
            f"Fix: delete '{persist_dir}' and re-run ingest from scratch, or "
            f"point --persist-dir at a new location."
        )

    return identifier


def build_index(chunks: list[Document], embedding_fn: Embeddings, persist_dir: str) -> Chroma:
    """Embed and persist a fresh Chroma collection from the given chunks."""
    identifier = _check_for_mismatch(persist_dir, embedding_fn)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=persist_dir,
        collection_name=settings.collection_name,
    )

    dimension = len(embedding_fn.embed_query("dimension probe"))
    _write_manifest(persist_dir, identifier, dimension)

    return vectorstore


def load_index(embedding_fn: Embeddings, persist_dir: str) -> Chroma:
    """Load an already-built Chroma collection from disk."""
    _check_for_mismatch(persist_dir, embedding_fn)

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_fn,
        collection_name=settings.collection_name,
    )
