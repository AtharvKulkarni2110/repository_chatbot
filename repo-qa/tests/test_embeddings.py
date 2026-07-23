import tempfile
from repo_qa.embeddings.provider import TfidfEmbeddings


def test_tfidf_embeddings_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = TfidfEmbeddings(tmpdir)
        vectors = embedder.embed_documents(["def login(): pass", "def logout(): pass"])

        assert len(vectors) == 2
        assert len(vectors[0]) == len(vectors[1]), "all vectors must share the same dimensionality"

        # A second instance loading from the same persist_dir should reuse
        # the exact vocabulary fit above, not fit a new one.
        reloaded = TfidfEmbeddings(tmpdir)
        query_vector = reloaded.embed_query("login")
        assert len(query_vector) == len(vectors[0])


def test_tfidf_embeddings_query_before_fit_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = TfidfEmbeddings(tmpdir)
        try:
            embedder.embed_query("test")
            assert False, "expected RuntimeError when querying before fitting"
        except RuntimeError:
            pass
