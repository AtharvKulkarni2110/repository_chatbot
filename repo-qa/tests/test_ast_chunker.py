from langchain_core.documents import Document
from repo_qa.ingestion.chunker import chunk_documents


PYTHON_SOURCE = '''
def add(a, b):
    return a + b


class Calculator:
    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        return a / b
'''


def test_python_functions_and_methods_become_separate_chunks():
    docs = [Document(page_content=PYTHON_SOURCE, metadata={"source": "calc.py"})]
    chunks = chunk_documents(docs)

    names = {c.metadata.get("function_name") for c in chunks}
    assert names == {"add", "multiply", "divide"}, (
        "expected one chunk per function/method, not per file or per class"
    )

    # Each chunk should contain ONLY its own function body, not the whole file.
    add_chunk = next(c for c in chunks if c.metadata["function_name"] == "add")
    assert "def add" in add_chunk.page_content
    assert "def multiply" not in add_chunk.page_content


def test_unsupported_language_falls_back_to_character_split():
    docs = [Document(page_content="some plain text\n\nwith no code structure at all",
                      metadata={"source": "notes.txt"})]
    chunks = chunk_documents(docs)

    assert len(chunks) >= 1
    assert chunks[0].metadata.get("function_name") is None, (
        "fallback chunks shouldn't have AST metadata"
    )


def test_file_with_no_functions_falls_back():
    # A Python file with no function/method definitions at all (e.g. just constants).
    docs = [Document(page_content="MAX_RETRIES = 3\nTIMEOUT = 30\n", metadata={"source": "constants.py"})]
    chunks = chunk_documents(docs)

    assert len(chunks) >= 1
    assert "MAX_RETRIES" in chunks[0].page_content


def test_chunk_index_still_assigned():
    docs = [Document(page_content=PYTHON_SOURCE, metadata={"source": "calc.py"})]
    chunks = chunk_documents(docs)

    for chunk in chunks:
        assert "chunk_index" in chunk.metadata
