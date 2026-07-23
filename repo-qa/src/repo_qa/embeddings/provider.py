"""
Embedding provider.

Tries a real sentence-transformer model (downloaded from Hugging Face on
first use, cached locally after that) for genuine semantic embeddings.

Falls back to a local TF-IDF embedding if the real model can't be reached
(no internet, locked-down network, etc.) so the rest of the pipeline can
still be exercised offline.

IMPORTANT: TF-IDF matches on shared vocabulary, not meaning. It's a
network-free stand-in for tests/demos, not a production embedding strategy.
See docs/concepts.md for why this distinction matters.
"""

import os
import pickle

from langchain_core.embeddings import Embeddings

from repo_qa.config import settings

TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"


def get_embedding_function(persist_dir: str, force_offline: bool = False) -> Embeddings:
    """Return an Embeddings instance: real model unless offline/unreachable."""
    if not force_offline:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            # trust_remote_code=True is required for jina-embeddings-v2-base-code
            # specifically - it ships custom modeling code on the HF Hub rather
            # than using a stock transformers architecture. Harmless no-op for
            # models that don't need it.
            model = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={"trust_remote_code": True},
            )
            model.embed_query("connectivity check")  # fail fast if the model can't download
            return model
        except Exception:
            # Deliberately broad: network errors, missing package, corrupted
            # cache, etc. should all fall back rather than crash ingestion.
            pass

    return TfidfEmbeddings(persist_dir)


class TfidfEmbeddings(Embeddings):
    """
    TF-IDF based embedding (offline fallback).

    The vocabulary is fit once, on the full set of chunks, at ingest time,
    then persisted to disk. query.py runs as a separate process later, so
    without persistence it would fit a *different* vocabulary on a
    single question and produce vectors that aren't comparable to the
    stored ones at all.
    """

    def __init__(self, persist_dir: str):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._persist_dir = persist_dir
        self._path = os.path.join(persist_dir, TFIDF_VECTORIZER_FILENAME)
        self._fitted = False

        if os.path.exists(self._path):
            with open(self._path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self._fitted = True
        else:
            self.vectorizer = TfidfVectorizer(max_features=settings.tfidf_max_features)

    def embed_documents(self, texts):
        vectors = self.vectorizer.fit_transform(texts)
        self._fitted = True
        os.makedirs(self._persist_dir, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        return vectors.toarray().tolist()

    def embed_query(self, text):
        if not self._fitted:
            raise RuntimeError("TF-IDF vectorizer not fit yet - run ingestion before querying.")
        return self.vectorizer.transform([text]).toarray()[0].tolist()
