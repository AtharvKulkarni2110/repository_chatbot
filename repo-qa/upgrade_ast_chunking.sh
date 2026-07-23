#!/usr/bin/env bash
# Upgrade script: AST-aware (tree-sitter) chunking + jina-embeddings-v2-base-code.
# Run this FROM INSIDE your existing repo-qa project directory
# (the one created by scaffold_repo_qa.sh).
#
# What this changes:
#   - src/repo_qa/ingestion/chunker.py   -> function/method-level chunks via tree-sitter
#   - src/repo_qa/config.py              -> default embedding model -> jina-embeddings-v2-base-code
#   - src/repo_qa/embeddings/provider.py -> passes trust_remote_code=True (jina needs this)
#   - requirements.txt, pyproject.toml   -> add tree-sitter-language-pack (maintained;
#     tree-sitter-languages is unmaintained and has NO wheels for Python 3.12+)
#   - tests/test_ast_chunker.py          -> new tests for the AST chunker (added file)
#
# After running: pip install -e . (to pick up new deps), then re-run ingest
# (old chroma_db dirs are NOT compatible - old runs used a different embedding
# model/dimensionality, so delete and rebuild the index).

set -e

if [ ! -f "pyproject.toml" ] || [ ! -d "src/repo_qa" ]; then
  echo "Error: run this from inside your repo-qa project root (pyproject.toml + src/repo_qa/ not found here)."
  exit 1
fi

echo "Applying AST chunking + embedding model upgrade..."

mkdir -p "src/repo_qa/ingestion"
cat > "src/repo_qa/ingestion/chunker.py" << 'UPGRADE_EOF'
"""
AST-aware document chunking (v1 — replaces v0's character-based splitting).

Chunks are drawn at function/method boundaries via tree-sitter, so a chunk
is always a complete, meaningful unit of code — matching how a human
actually reads code (as callable units), not an arbitrary character window
that can cut a function in half.

Strategy per file:
1. If the file's language has a tree-sitter grammar available, parse it and
   extract every function/method definition as its own chunk.
2. If a single extracted unit is unusually large (e.g. a huge function),
   sub-split it with the old character-based splitter so it doesn't exceed
   what the embedding model can meaningfully encode.
3. If the language isn't supported, or the file has no functions/methods at
   all (e.g. a config file, a dataclass with no methods, a markdown doc),
   fall back to the old character-based splitter for that whole file.

This means ingestion never hard-fails on an unsupported file type — it just
degrades to v0 behavior for that one file.

Known limitation (intentional, for now): a method chunk doesn't carry its
enclosing class name in its text or metadata yet, so "class Foo: def
bar()" shows up as a chunk for just `bar`, without "Foo" for context. Fine
for most single-purpose methods; a planned follow-up is to attach
`parent_class` metadata by tracking ancestry while walking the tree.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from repo_qa.config import settings

# Maps file extension -> (tree-sitter language name, node types treated as
# a "chunkable unit"). Deliberately using the smallest meaningful callable
# unit (function/method), not class/module, so a chunk is never bigger than
# it needs to be. A class with no methods (e.g. a plain dataclass) simply
# won't match anything here and falls through to the character splitter.
LANGUAGE_MAP = {
    ".py": ("python", {"function_definition"}),
    ".js": ("javascript", {"function_declaration", "method_definition"}),
    ".jsx": ("javascript", {"function_declaration", "method_definition"}),
    ".ts": ("typescript", {"function_declaration", "method_definition"}),
    ".tsx": ("tsx", {"function_declaration", "method_definition"}),
    ".java": ("java", {"method_declaration"}),
    ".go": ("go", {"function_declaration", "method_declaration"}),
    ".rb": ("ruby", {"method"}),
    ".c": ("c", {"function_definition"}),
    ".cpp": ("cpp", {"function_definition"}),
    ".rs": ("rust", {"function_item"}),
}

# If an extracted AST chunk is bigger than this, sub-split it further.
# Generous multiple of chunk_size: most functions are fine as one chunk;
# this only kicks in for genuinely oversized functions, so we don't
# silently exceed what the embedding model can encode in one pass.
MAX_AST_CHUNK_CHARS_MULTIPLIER = 3

_parser_cache: dict[str, object] = {}


def _get_parser(language_name: str):
    """Load and cache a tree-sitter parser for the given language."""
    if language_name not in _parser_cache:
        from tree_sitter_language_pack import get_parser
        _parser_cache[language_name] = get_parser(language_name)
    return _parser_cache[language_name]


def _language_for_source(source_path: str):
    """Look up (language_name, unit_types) for a file path, or None if unsupported."""
    for ext, config in LANGUAGE_MAP.items():
        if source_path.endswith(ext):
            return config
    return None


def _extract_ast_chunks(doc: Document, language_name: str, unit_types: set) -> list[Document] | None:
    """
    Try to split one document into function/method-level chunks.
    Returns None if the language can't be parsed at all (missing grammar,
    parser error) so the caller knows to fall back - as opposed to an empty
    list, which means "parsed fine, just no matching units in this file".
    """
    try:
        parser = _get_parser(language_name)
    except Exception:
        return None

    try:
        tree = parser.parse(doc.page_content.encode("utf-8"))
    except Exception:
        return None

    chunks: list[Document] = []

    def walk(node):
        if node.type in unit_types:
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf-8", errors="ignore") if name_node else None
            chunks.append(Document(
                page_content=node.text.decode("utf-8", errors="ignore"),
                metadata={
                    **doc.metadata,
                    "node_type": node.type,
                    "function_name": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                },
            ))
            return  # don't descend further - keeps the outer unit as one chunk, not split into nested pieces
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks


def _character_split(doc: Document) -> list[Document]:
    """The v0 fallback splitter, used for unsupported languages and oversized chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents([doc])


def _split_oversized_chunk(chunk: Document) -> list[Document]:
    """If an AST-extracted chunk is unusually large, sub-split it further."""
    max_chars = settings.chunk_size * MAX_AST_CHUNK_CHARS_MULTIPLIER
    if len(chunk.page_content) <= max_chars:
        return [chunk]
    return _character_split(chunk)


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents into chunks, preferring function/method-boundary chunks
    via tree-sitter, falling back to character-based splitting per-file
    when the language is unsupported or the file has no matching units.
    """
    all_chunks: list[Document] = []

    for doc in docs:
        source = doc.metadata.get("source", "")
        lang_config = _language_for_source(source)

        ast_chunks = None
        if lang_config:
            language_name, unit_types = lang_config
            ast_chunks = _extract_ast_chunks(doc, language_name, unit_types)

        if ast_chunks:  # non-empty list of real function/method chunks
            for chunk in ast_chunks:
                all_chunks.extend(_split_oversized_chunk(chunk))
        else:  # ast_chunks is None (unsupported language) or [] (no functions found)
            all_chunks.extend(_character_split(doc))

    # Tag each chunk with its position within its source file, for citations.
    counters: dict[str, int] = {}
    for chunk in all_chunks:
        src = chunk.metadata["source"]
        counters[src] = counters.get(src, 0) + 1
        chunk.metadata["chunk_index"] = counters[src]

    return all_chunks
UPGRADE_EOF
echo "  updated: src/repo_qa/ingestion/chunker.py"

mkdir -p "src/repo_qa"
cat > "src/repo_qa/config.py" << 'UPGRADE_EOF'
"""
Central configuration for repo_qa.

Why this file exists: in the v0 scripts, things like chunk size, ignored
directories, and model names were hardcoded inline wherever they were used.
That's fine for a 100-line script, but it doesn't scale — you end up
changing the same constant in three different files and missing one.

Everything tunable lives here. Values can be overridden via environment
variables (useful for prod: different chunk sizes per environment, swapping
models without touching code, etc.) without editing this file.
"""

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- Ingestion ---
    ignore_dirs: frozenset = field(default_factory=lambda: frozenset({
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".next", ".idea", ".vscode", "chroma_db",
    }))
    include_extensions: frozenset = field(default_factory=lambda: frozenset({
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb",
        ".md", ".txt", ".json", ".yaml", ".yml", ".rs", ".c", ".cpp", ".h",
    }))
    max_file_size_bytes: int = _env_int("REPO_QA_MAX_FILE_SIZE", 500_000)

    # --- Chunking ---
    chunk_size: int = _env_int("REPO_QA_CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("REPO_QA_CHUNK_OVERLAP", 100)

    # --- Embeddings ---
    # jina-embeddings-v2-base-code: purpose-trained on GitHub code across
    # ~30 languages (~161M params). Chosen over general-purpose models like
    # all-MiniLM-L6-v2 specifically because this project embeds code, not
    # prose - a code-trained model should place semantically similar
    # functions closer together than a model that's never seen code syntax.
    embedding_model_name: str = _env_str(
        "REPO_QA_EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-code"
    )
    tfidf_max_features: int = _env_int("REPO_QA_TFIDF_MAX_FEATURES", 4096)

    # --- Vector store ---
    default_persist_dir: str = _env_str("REPO_QA_PERSIST_DIR", "./chroma_db")
    collection_name: str = _env_str("REPO_QA_COLLECTION", "repo_chunks")

    # --- Retrieval ---
    default_top_k: int = _env_int("REPO_QA_TOP_K", 5)

    # --- Generation ---
    llm_model_name: str = _env_str("REPO_QA_LLM_MODEL", "claude-sonnet-4-6")
    llm_max_tokens: int = _env_int("REPO_QA_LLM_MAX_TOKENS", 1000)


settings = Settings()
UPGRADE_EOF
echo "  updated: src/repo_qa/config.py"

mkdir -p "src/repo_qa/embeddings"
cat > "src/repo_qa/embeddings/provider.py" << 'UPGRADE_EOF'
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
UPGRADE_EOF
echo "  updated: src/repo_qa/embeddings/provider.py"

cat > "requirements.txt" << 'UPGRADE_EOF'
langchain
langchain-community
langchain-chroma
langchain-huggingface
langchain-anthropic
chromadb
scikit-learn
tree-sitter-language-pack
UPGRADE_EOF
echo "  updated: requirements.txt"

cat > "pyproject.toml" << 'UPGRADE_EOF'
[project]
name = "repo-qa"
version = "0.1.0"
description = "Read-only Q&A over a local codebase using LangChain RAG"
requires-python = ">=3.10"
dependencies = [
    "langchain",
    "langchain-community",
    "langchain-chroma",
    "langchain-huggingface",
    "langchain-anthropic",
    "chromadb",
    "scikit-learn",
    "tree-sitter-language-pack",
]

[project.scripts]
repo-qa = "repo_qa.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
UPGRADE_EOF
echo "  updated: pyproject.toml"

mkdir -p "tests"
cat > "tests/test_ast_chunker.py" << 'UPGRADE_EOF'
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
UPGRADE_EOF
echo "  updated: tests/test_ast_chunker.py"

echo ""
echo "Done. Next steps:"
echo "  pip install -e .                          # picks up tree-sitter-language-pack"
echo "  rm -rf chroma_db                          # old index used a different embedding model - must rebuild"
echo "  repo-qa ingest sample_repo --persist-dir ./chroma_db"
echo "  repo-qa query \"how long is a session token valid for?\" --persist-dir ./chroma_db"
echo ""
echo "Add --offline to ingest/query if jina-embeddings-v2-base-code cannot download"
echo "(e.g. no internet to huggingface.co) - falls back to TF-IDF automatically either way."
