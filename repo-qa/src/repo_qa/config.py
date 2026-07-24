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
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


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
    # Provider is swappable: "google" (Gemini, free tier available) or
    # "anthropic" (Claude). Free-tier boundaries on any provider shift often -
    # verify current pricing/quotas before relying on this long-term.
    # Google: https://ai.google.dev/pricing | Anthropic: https://www.anthropic.com/pricing
    llm_provider: str = _env_str("REPO_QA_LLM_PROVIDER", "google")
    llm_model_name: str = _env_str("REPO_QA_LLM_MODEL", "gemini-2.0-flash")
    llm_max_tokens: int = _env_int("REPO_QA_LLM_MAX_TOKENS", 1000)


settings = Settings()
