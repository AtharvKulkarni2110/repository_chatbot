"""
File discovery and loading.

Responsibility: turn a folder on disk into a list of LangChain Document
objects. Nothing about chunking or embedding happens here — this module's
only job is "find the right files and read them safely".
"""

import os
from langchain_core.documents import Document

from repo_qa.config import settings


def collect_files(repo_path: str) -> list[str]:
    """Walk the repo and return absolute paths of files worth indexing."""
    collected = []
    for root, dirs, files in os.walk(repo_path):
        # Prune ignored directories in place so os.walk skips descending
        # into them at all (much faster than filtering after the fact on
        # a large repo with e.g. node_modules).
        dirs[:] = [d for d in dirs if d not in settings.ignore_dirs]

        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext not in settings.include_extensions:
                continue
            full_path = os.path.join(root, filename)
            if os.path.getsize(full_path) > settings.max_file_size_bytes:
                continue
            collected.append(full_path)
    return collected


def load_documents(repo_path: str, file_paths: list[str]) -> list[Document]:
    """Read each file into a Document, tagged with its path relative to the repo root."""
    docs = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            # Unreadable file (permissions, broken symlink, etc.) - skip rather than crash the whole ingest run.
            continue

        if not text.strip():
            continue

        rel_path = os.path.relpath(path, repo_path)
        docs.append(Document(page_content=text, metadata={"source": rel_path}))
    return docs


def load_repo(repo_path: str) -> list[Document]:
    """Convenience wrapper: collect + load in one call."""
    repo_path = os.path.abspath(repo_path)
    file_paths = collect_files(repo_path)
    return load_documents(repo_path, file_paths)
