from langchain_core.documents import Document
from repo_qa.ingestion.chunker import chunk_documents


def test_chunk_documents_tags_chunk_index():
    docs = [Document(page_content="a" * 2000, metadata={"source": "foo.py"})]
    chunks = chunk_documents(docs)

    assert len(chunks) > 1, "a 2000-char doc should split into multiple chunks at default chunk_size"
    for chunk in chunks:
        assert chunk.metadata["source"] == "foo.py"
        assert "chunk_index" in chunk.metadata


def test_chunk_documents_indexes_restart_per_file():
    docs = [
        Document(page_content="a" * 2000, metadata={"source": "foo.py"}),
        Document(page_content="b" * 2000, metadata={"source": "bar.py"}),
    ]
    chunks = chunk_documents(docs)

    foo_chunks = [c for c in chunks if c.metadata["source"] == "foo.py"]
    bar_chunks = [c for c in chunks if c.metadata["source"] == "bar.py"]

    assert foo_chunks[0].metadata["chunk_index"] == 1
    assert bar_chunks[0].metadata["chunk_index"] == 1
