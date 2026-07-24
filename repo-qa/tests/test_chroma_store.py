import tempfile
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from repo_qa.vectorstore.chroma_store import build_index, load_index, EmbeddingMismatchError


class FakeEmbeddings(Embeddings):
    """A minimal fake embedding model, standing in for e.g. jina vs TF-IDF,
    without needing real network access or a real model to demonstrate the guard."""

    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim

    def embed_documents(self, texts):
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.1] * self.dim


def _sample_docs():
    return [Document(page_content="def foo(): pass", metadata={"source": "a.py", "chunk_index": 1})]


def test_build_then_build_with_different_model_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        build_index(_sample_docs(), FakeEmbeddings("model-a", dim=4), tmpdir)

        with pytest.raises(EmbeddingMismatchError, match="model-a.*model-b"):
            build_index(_sample_docs(), FakeEmbeddings("model-b", dim=4), tmpdir)


def test_build_then_load_with_different_model_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        build_index(_sample_docs(), FakeEmbeddings("jinaai/jina-embeddings-v2-base-code", dim=768), tmpdir)

        with pytest.raises(EmbeddingMismatchError):
            load_index(FakeEmbeddings("tfidf-offline", dim=340), tmpdir)


def test_build_then_load_with_same_model_succeeds():
    with tempfile.TemporaryDirectory() as tmpdir:
        embedding_fn = FakeEmbeddings("model-a", dim=4)
        build_index(_sample_docs(), embedding_fn, tmpdir)

        # should not raise
        load_index(FakeEmbeddings("model-a", dim=4), tmpdir)


def test_fresh_persist_dir_has_no_manifest_conflict():
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        empty_dir = os.path.join(tmpdir, "fresh")
        # should not raise - nothing built here yet
        build_index(_sample_docs(), FakeEmbeddings("any-model", dim=4), empty_dir)
