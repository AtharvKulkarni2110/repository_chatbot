#!/usr/bin/env bash
# Scaffold script for the repo-qa project.
# Generates the full directory structure + files in the current directory.
# Usage: bash scaffold_repo_qa.sh [target_dir]

set -e

TARGET="${1:-repo-qa}"
mkdir -p "$TARGET"
cd "$TARGET"

echo "Creating project in: $(pwd)"

# --- directories ---
mkdir -p "docs"
mkdir -p "sample_repo"
mkdir -p "sample_repo/auth"
mkdir -p "sample_repo/db"
mkdir -p "sample_repo/utils"
mkdir -p "src"
mkdir -p "src/repo_qa"
mkdir -p "src/repo_qa/embeddings"
mkdir -p "src/repo_qa/generation"
mkdir -p "src/repo_qa/ingestion"
mkdir -p "src/repo_qa/retrieval"
mkdir -p "src/repo_qa/vectorstore"
mkdir -p "tests"

# --- files ---
cat > ".env.example" << 'REPO_QA_EOF'
# Copy this file to .env and fill in real values (never commit the real .env)

# Required for the answer-generation step. Without this, the CLI still
# works but only shows retrieved chunks instead of a generated answer.
ANTHROPIC_API_KEY=sk-ant-...

# --- Optional overrides (defaults live in src/repo_qa/config.py) ---
# REPO_QA_CHUNK_SIZE=800
# REPO_QA_CHUNK_OVERLAP=100
# REPO_QA_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# REPO_QA_PERSIST_DIR=./chroma_db
# REPO_QA_TOP_K=5
# REPO_QA_LLM_MODEL=claude-sonnet-4-6
REPO_QA_EOF

cat > ".gitignore" << 'REPO_QA_EOF'
chroma_db/
*.pkl
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
.env
.DS_Store
REPO_QA_EOF

cat > "README.md" << 'REPO_QA_EOF'
# repo-qa

Read-only Q&A over a local codebase, using a LangChain RAG (Retrieval-Augmented
Generation) pipeline. Point it at any repo, ask questions in plain English,
get answers grounded in the actual code with source citations.

This is **Stage 1** of a larger "AI software engineer" project (see
`docs/concepts.md` for the full roadmap). This stage is read-only: it
answers questions, it does not edit files, run tests, or make commits.

## Quick start

```bash
git clone <this repo>
cd repo-qa
pip install -e .
cp .env.example .env   # then fill in ANTHROPIC_API_KEY

# Index a repo (run once, re-run whenever the code changes materially)
repo-qa ingest sample_repo --persist-dir ./chroma_db

# Ask questions
repo-qa query "how long is a session token valid for?" --persist-dir ./chroma_db
```

Without `ANTHROPIC_API_KEY` set, `repo-qa query` still runs and prints the
retrieved chunks — useful for judging retrieval quality on its own, without
paying for an LLM call every time.

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/requirements.md`](docs/requirements.md) | Functional/non-functional requirements, dependencies, system requirements |
| [`docs/file_structure.md`](docs/file_structure.md) | What every file/folder does and why it's organized this way |
| [`docs/concepts.md`](docs/concepts.md) | The theory: RAG, embeddings, vector search, chunking, and every term used in this project, explained |

## Project layout (short version)

```
src/repo_qa/
├── config.py       # all tunable settings, one place
├── cli.py          # entry point: repo-qa ingest / repo-qa query
├── ingestion/       # walk repo -> load files -> chunk
├── embeddings/       # turn text into vectors
├── vectorstore/       # persist/load the vector index (Chroma)
├── retrieval/       # top-k similarity search
└── generation/       # prompt + LLM call
```

Full explanation with rationale: `docs/file_structure.md`.

## Known limitation (intentional, for now)

Chunking is character-based (`RecursiveCharacterTextSplitter`), so a
function can get split across two chunks. This is called out because it's
the single biggest lever on answer quality right now — the next planned
improvement is tree-sitter/AST-aware chunking (see `docs/concepts.md`,
"Planned Improvements").

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```
REPO_QA_EOF

cat > "docs/concepts.md" << 'REPO_QA_EOF'
# Concepts & Theory

Every concept here is tied back to a specific file in this project, so you
can read the explanation and immediately go look at the code that implements
it. Read this top to bottom once — it follows the actual data flow through
the pipeline.

---

## 1. The big picture: what is RAG?

**RAG = Retrieval-Augmented Generation.**

An LLM only knows what was in its training data, plus whatever text you put
directly in the prompt. It has never seen your repo. RAG is the pattern of:

1. Break your data into searchable pieces ahead of time (**indexing**)
2. At question time, find the pieces relevant to the question (**retrieval**)
3. Hand only those pieces to the LLM as context, and ask it to answer using
   them (**generation**)

This is *not* fine-tuning. Nothing about the LLM itself changes. You're just
being selective about what goes into the prompt, because you can't fit an
entire codebase into one prompt.

In this project: `ingestion/` + `embeddings/` + `vectorstore/` do step 1,
`retrieval/` does step 2, `generation/` does step 3. `cli.py` wires them
together in that order.

---

## 2. Embeddings

An **embedding** is a list of numbers (a vector) representing the *meaning*
of a piece of text. Texts with similar meaning get vectors that are close
together in that numeric space; unrelated texts get vectors far apart.

Example (simplified to 2 dimensions for intuition; real models use hundreds):
```
"create a session token"  -> [0.81, 0.42]
"generate an auth token"  -> [0.79, 0.45]   <- close to the above (similar meaning)
"connect to the database" -> [0.10, 0.88]   <- far from both (different meaning)
```

**Where this happens in the project:** `embeddings/provider.py`.

We use `sentence-transformers/all-MiniLM-L6-v2` when it can be downloaded —
a small model specifically trained so that semantically similar sentences
end up with similar vectors. This is what lets you ask "where is auth
handled?" and match code that never uses the literal word "auth".

### The offline fallback: TF-IDF (and why it's *not* the same thing)

TF-IDF (**Term Frequency–Inverse Document Frequency**) is a much older,
purely statistical technique. It scores a word as important to a document
if it appears often *there* but rarely *elsewhere* in the corpus. It has no
concept of meaning — `"token"` and `"credential"` are just as unrelated to
it as `"token"` and `"banana"`.

This project uses TF-IDF (`TfidfEmbeddings` in `embeddings/provider.py`)
only as a network-free fallback so the pipeline can be tested without
internet access. It's included deliberately as a teaching contrast: once you
have both working, try the same question against each and see the
difference in retrieved chunks. That difference *is* what "semantic search"
means in practice.

### Why the vectorizer gets pickled to disk

`ingest` and `query` run as **separate processes** (you run one command,
then later another). A `TfidfVectorizer`'s vocabulary is built from
whatever text it's fit on — if `query` fit its own vectorizer on just the
one question, it would produce vectors in a totally different, incompatible
numeric space from the ones stored during `ingest`. So the vectorizer
itself is saved (`tfidf_vectorizer.pkl`) and reloaded, not rebuilt. Real
embedding models don't have this problem — they're pre-trained once, not
fit per-run, so the same model always produces comparable vectors.

---

## 3. Vector databases & similarity search

A **vector database** stores embeddings and answers the question "which
stored vectors are closest to this query vector?" efficiently — this
project uses **Chroma**, which runs locally with no separate server needed
(`vectorstore/chroma_store.py`).

"Closest" is measured by a **distance metric**. Two common ones:
- **Cosine similarity** — measures the *angle* between two vectors, ignoring
  their length. Most common for text embeddings.
- **Euclidean distance** — straight-line distance between two points.

**Where this shows up in the output you saw:** `retrieval/retriever.py`
returns `(chunk, distance)` pairs where *lower distance = more similar*.
This is the opposite direction from a "relevance score" (where higher =
better) — a common source of bugs if you mix the two up, which is exactly
what the earlier warning about "relevance scores must be between 0 and 1"
was flagging. We use raw distance explicitly to avoid that confusion.

**Top-k retrieval** just means "give me the k closest vectors" — in this
project, `k` defaults to 5 (`settings.default_top_k` in `config.py`),
overridable with `-k` on the CLI.

---

## 4. Chunking

You can't embed an entire file as one vector and expect it to usefully
match a specific question about one function buried inside it — the vector
would represent the "average meaning" of the whole file, diluting the
specific part you care about. So files are split into smaller **chunks**
before embedding.

**Where this happens:** `ingestion/chunker.py`, using LangChain's
`RecursiveCharacterTextSplitter`.

This splitter tries a list of separators in order (`\n\n`, then `\n`, then
` `, then raw characters) and picks the largest natural break point it can
find under the `chunk_size` limit. It's "recursive" in that if splitting on
paragraphs still leaves a chunk too big, it recurses down to splitting on
single newlines, then words, then characters.

**`chunk_overlap`** (default 100 characters) means consecutive chunks share
a bit of text at the boundary. This helps when something meaningful sits
right at a chunk boundary — without overlap, it could get cut such that
neither chunk fully contains it.

### Why this is called out as the current weak point

This splitter has zero awareness of code structure. It doesn't know what a
function is — it just counts characters. It's entirely possible for a
function's signature to end up in one chunk and its body in the next,
splitting something that should be retrieved as one coherent unit.

**The fix (not yet built): AST-aware chunking.** An AST (Abstract Syntax
Tree) is a structured representation of code's actual grammar — "this is a
function definition, this is its name, this is its body" — rather than
raw text. A tool like **tree-sitter** parses code into this structure so
chunks can be drawn at function/class boundaries instead of arbitrary
character counts. This is the single highest-leverage improvement available
to this project right now, which is why it's first in the roadmap below.

---

## 5. Grounding & prompting

**Where this happens:** `generation/answerer.py`, `ANSWER_PROMPT_TEMPLATE`.

"Grounding" means constraining an LLM's answer to only use information
you've explicitly given it, rather than its general training knowledge —
which for your specific, private repo, it never saw. The prompt does this
with two instructions:

1. *"using ONLY the context below"* — the core grounding instruction.
2. *"If the answer isn't in the context, say so explicitly instead of
   guessing"* — without this, LLMs tend to confidently fabricate a plausible
   -sounding answer rather than admit the retrieved chunks didn't cover it.
   This failure mode (fabricating a confident but false answer) is usually
   called **hallucination**.

The retrieved chunks are formatted with their file path and chunk number
attached (`format_context()`) specifically so the LLM can cite sources —
this is what FR-5 in `docs/requirements.md` requires, and it's also what
lets *you* verify an answer instead of trusting it blindly.

---

## 6. Why retrieval and generation are separate steps you can inspect independently

A subtlety worth internalizing: **most bad RAG answers are retrieval
failures wearing an LLM costume.** If the wrong chunks get retrieved, the
LLM will (correctly, per its instructions) either answer based on those
wrong chunks or say "not in context" — either way, the *retrieval* step is
where the fix belongs, not the prompt.

This is why `repo-qa query` still prints retrieved chunks even without an
`ANTHROPIC_API_KEY` (`generation/answerer.py`, `llm_available()`) — so you
can evaluate retrieval quality on its own, without an LLM call in the way,
before ever debugging the generation step.

---

## 7. LangChain's role, concretely

LangChain isn't doing anything magic — it's providing standardized
interfaces so the pieces above snap together:

- `Document` — a standard container for `(text, metadata)`, used everywhere
  from `loader.py` through to `answerer.py`.
- `RecursiveCharacterTextSplitter` — the chunker (see §4).
- `Embeddings` (base class) — the interface both the real HuggingFace model
  and our custom `TfidfEmbeddings` implement, so `vectorstore/` doesn't
  need to know or care which one it's talking to.
- `Chroma` (via `langchain-chroma`) — a wrapper so vector store calls look
  the same regardless of which underlying database you pick.
- `ChatAnthropic` (via `langchain-anthropic`) — a standard chat-model
  interface, so swapping to a different LLM provider later would mean
  changing an import, not rewriting `answerer.py`.

The practical value: every module in this project could have its
implementation swapped (different chunker, different vector DB, different
LLM) without touching the modules around it, because they all talk to each
other through these shared interfaces rather than to concrete SDKs directly.

---

## 8. Planned improvements (roadmap, ordered by leverage)

1. **AST-aware chunking** (tree-sitter) — replace `chunker.py`'s character
   splitting with function/class-boundary splitting. Highest expected
   impact on answer quality; described in §4.
2. **Metadata filtering** — use the `source`/`chunk_index` metadata already
   being stored to let retrieval filter by file type or folder.
3. **Hybrid search** — combine the current vector similarity search with
   keyword/BM25 search, so exact identifiers (variable names, error codes)
   that don't have strong "semantic" meaning still match reliably.
4. **Re-ranking** — retrieve a larger candidate set cheaply, then use a
   more expensive but more accurate model to re-order just the top
   candidates before they reach the LLM.
5. **Agent loop** — the biggest architectural jump. Currently this is a
   fixed, single-pass pipeline: retrieve once, answer once. An agent
   instead gives the LLM tools (`search_code`, `read_file`, etc.) and lets
   it decide, iteratively, what to look up next — the foundation needed
   before this project could ever safely *edit* code rather than just
   describe it.
REPO_QA_EOF

cat > "docs/file_structure.md" << 'REPO_QA_EOF'
# File Structure

```
repo-qa/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── docs/
│   ├── requirements.md
│   ├── file_structure.md      <- you are here
│   └── concepts.md
├── src/
│   └── repo_qa/
│       ├── __init__.py
│       ├── config.py
│       ├── cli.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── chunker.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── provider.py
│       ├── vectorstore/
│       │   ├── __init__.py
│       │   └── chroma_store.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   └── retriever.py
│       └── generation/
│           ├── __init__.py
│           └── answerer.py
├── sample_repo/                # tiny fake codebase to test against
│   ├── auth/login.py
│   ├── db/connection.py
│   └── utils/logging.py
└── tests/
    ├── __init__.py
    ├── test_chunker.py
    └── test_embeddings.py
```

## Why `src/repo_qa/` instead of just `repo_qa/` at the root?

This is the "src layout" — a standard Python packaging pattern. Without the
`src/` folder, if you run tests from the project root, Python can silently
import your *uncommitted, unpacked* source directory instead of the actually
*installed* package, hiding packaging bugs. The `src/` layout forces you to
`pip install -e .` before anything works, which means what you test is what
would actually get installed by someone else. Small thing, but it's a real
source of "works on my machine" bugs in real projects.

## Why is the pipeline split into 5 separate folders instead of one `pipeline.py`?

Each folder is one stage of the RAG pipeline, and each stage has a single
responsibility:

| Folder | Single responsibility | Why isolated |
|---|---|---|
| `ingestion/` | Turn files on disk into chunks of text | Pure I/O + text processing, no ML, easy to unit test without any model |
| `embeddings/` | Turn text into vectors | The one place that talks to an embedding model — swap models here only |
| `vectorstore/` | Persist/load vectors | The one place that talks to Chroma — swap databases here only |
| `retrieval/` | Given a question, return top-k chunks | Testable independent of whether an LLM key exists |
| `generation/` | Given chunks + question, produce an answer | The one place that talks to an LLM — the only "expensive" call in the system |

The practical benefit: if retrieval quality is bad, you look in exactly one
folder (`retrieval/`, or possibly `embeddings/`), not through one 300-line
script trying to figure out which of five things went wrong. It also means
each of these can be swapped independently later — e.g. replacing Chroma
with a different vector DB touches only `vectorstore/chroma_store.py`.

## What each file does

- **`config.py`** — every tunable value (chunk size, model names, default
  paths) in one place, overridable via environment variables. No magic
  numbers hardcoded elsewhere in the codebase.
- **`cli.py`** — the entry point. Parses command-line args and calls the
  pipeline modules in order. Deliberately "thin" — it contains no logic of
  its own, only orchestration, so it stays readable as a map of how the
  pieces fit together.
- **`ingestion/loader.py`** — walks the repo, filters out junk
  directories/files, reads the rest into LangChain `Document` objects.
- **`ingestion/chunker.py`** — splits documents into overlapping chunks.
  Isolated specifically because this is the piece planned to be swapped for
  tree-sitter/AST-aware chunking next (see `docs/concepts.md`).
- **`embeddings/provider.py`** — returns an embedding function: real
  semantic embeddings when reachable, TF-IDF fallback when offline.
- **`vectorstore/chroma_store.py`** — build/load a Chroma collection. The
  only file that imports `langchain_chroma` directly.
- **`retrieval/retriever.py`** — runs a similarity search against a loaded
  vector store, returns ranked chunks.
- **`generation/answerer.py`** — builds the prompt from retrieved chunks
  and calls the LLM. The only file that imports `langchain_anthropic`.
- **`tests/`** — unit tests for the modules that don't require network
  access to test meaningfully (chunker, offline embeddings).
- **`sample_repo/`** — a tiny fake codebase (auth/db/utils modules) so the
  pipeline can be tried immediately without pointing it at a real project.

## Data that gets generated (not committed — see `.gitignore`)

- `chroma_db/` — the persisted vector index, created by `repo-qa ingest`.
  Contains embedded chunks and, if using offline mode, the pickled TF-IDF
  vectorizer (`tfidf_vectorizer.pkl`).
REPO_QA_EOF

cat > "docs/requirements.md" << 'REPO_QA_EOF'
# Requirements

This doc covers two different meanings of "requirements": what the system
needs to do (functional), and what it needs to run (technical/dependencies).
Conflating these is a common source of confusion in project docs, so they're
kept clearly separate here.

## 1. Functional requirements (what it must do)

| ID | Requirement |
|---|---|
| FR-1 | Given a path to a local repo, index its text/code files for search. |
| FR-2 | Skip non-code artifacts (binaries, `.git`, `node_modules`, build output) automatically. |
| FR-3 | Accept a natural-language question and return the most relevant code chunks. |
| FR-4 | Generate a natural-language answer grounded only in retrieved chunks (no hallucinated APIs/behavior). |
| FR-5 | Cite which file(s) an answer is based on. |
| FR-6 | Re-indexing must be possible without manual cleanup (re-running `ingest` should just work). |
| FR-7 | The system must be inspectable at each stage — a user can see retrieved chunks even without an LLM call, to separate "is retrieval good" from "is the LLM answer good". |

## 2. Non-functional requirements (how well it must do it)

| ID | Requirement | Why |
|---|---|---|
| NFR-1 | Must run fully locally except for the two external calls (embedding model download, LLM call). | Avoids sending an entire codebase to a third party by default. |
| NFR-2 | Must degrade gracefully without internet access. | The offline TF-IDF fallback exists so ingestion/retrieval can be tested/demoed without depending on Hugging Face being reachable. |
| NFR-3 | Must degrade gracefully without an LLM API key. | Retrieval and generation are decoupled — one working without the other is a feature, not a bug, during development. |
| NFR-4 | Config must be centralized, not scattered across files. | See `src/repo_qa/config.py` — a single change (e.g. chunk size) shouldn't require hunting through multiple files. |
| NFR-5 | Modules must be independently testable. | Ingestion, embeddings, retrieval, and generation are separate modules specifically so each can be unit tested without needing the others (see `tests/`). |

## 3. System requirements

- Python 3.10+
- ~200MB disk for the embedding model cache (first run only, if using real embeddings)
- Internet access to `huggingface.co` for the first embedding-model download (optional — TF-IDF fallback works fully offline)
- An Anthropic API key for the answer-generation step (optional — retrieval-only mode works without one)

## 4. Dependencies and why each one is here

| Package | Role |
|---|---|
| `langchain`, `langchain-community` | Core abstractions: `Document`, text splitters, chain interfaces. |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` used for chunking. |
| `langchain-chroma` | LangChain's wrapper around the Chroma vector database. |
| `chromadb` | The actual vector database — stores embeddings, does similarity search. |
| `langchain-huggingface` | Loads `sentence-transformers/all-MiniLM-L6-v2` for real semantic embeddings. |
| `scikit-learn` | Only used for `TfidfVectorizer` — the offline embedding fallback. |
| `langchain-anthropic` | Calls Claude for the answer-generation step. |
| `pytest` (dev only) | Runs `tests/`. |

## 5. Out of scope for this stage (see `docs/concepts.md` roadmap)

- Writing/editing files
- Running tests or code
- Multi-file / cross-file reasoning
- Any form of agent loop or planning (this is a single-pass RAG pipeline, not an agent)
- MCP servers, Docker sandboxing, Git integration
REPO_QA_EOF

cat > "pyproject.toml" << 'REPO_QA_EOF'
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
]

[project.scripts]
repo-qa = "repo_qa.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
REPO_QA_EOF

cat > "requirements.txt" << 'REPO_QA_EOF'
langchain
langchain-community
langchain-chroma
langchain-huggingface
langchain-anthropic
chromadb
scikit-learn
REPO_QA_EOF

cat > "sample_repo/auth/login.py" << 'REPO_QA_EOF'
"""Authentication module: handles user login and session tokens."""

import hashlib
import secrets
from datetime import datetime, timedelta

SESSION_STORE = {}


def hash_password(password: str, salt: str) -> str:
    """Hash a password with a salt using SHA-256."""
    return hashlib.sha256((password + salt).encode()).hexdigest()


def create_session_token(user_id: str) -> str:
    """Generate a random session token and store it with an expiry."""
    token = secrets.token_hex(32)
    SESSION_STORE[token] = {
        "user_id": user_id,
        "expires_at": datetime.utcnow() + timedelta(hours=24),
    }
    return token


def login(username: str, password: str, db_lookup_fn) -> str:
    """
    Validate credentials against the database and return a session token.
    Raises ValueError if credentials are invalid.
    """
    user = db_lookup_fn(username)
    if user is None:
        raise ValueError("User not found")

    hashed = hash_password(password, user["salt"])
    if hashed != user["password_hash"]:
        raise ValueError("Invalid password")

    return create_session_token(user["id"])


def validate_session(token: str) -> bool:
    """Check whether a session token is present and not expired."""
    session = SESSION_STORE.get(token)
    if session is None:
        return False
    return datetime.utcnow() < session["expires_at"]


def logout(token: str) -> None:
    """Invalidate a session token."""
    SESSION_STORE.pop(token, None)
REPO_QA_EOF

cat > "sample_repo/db/connection.py" << 'REPO_QA_EOF'
"""Database connection and user lookup utilities."""

import sqlite3
from contextlib import contextmanager

DB_PATH = "app.db"


@contextmanager
def get_connection():
    """Open a SQLite connection and ensure it closes cleanly."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_schema():
    """Create the users table if it doesn't already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def find_user_by_username(username: str):
    """Look up a single user row by username, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def insert_user(user_id: str, username: str, password_hash: str, salt: str, created_at: str):
    """Insert a new user row into the users table."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, password_hash, salt, created_at),
        )
        conn.commit()
REPO_QA_EOF

cat > "sample_repo/utils/logging.py" << 'REPO_QA_EOF'
"""Simple structured logging helper used across the app."""

import json
import sys
from datetime import datetime


def log_event(level: str, message: str, **fields):
    """Print a JSON log line to stdout with a timestamp and level."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level.upper(),
        "message": message,
        **fields,
    }
    print(json.dumps(record), file=sys.stdout)


def log_info(message: str, **fields):
    log_event("info", message, **fields)


def log_error(message: str, **fields):
    log_event("error", message, **fields)
REPO_QA_EOF

cat > "src/repo_qa/__init__.py" << 'REPO_QA_EOF'

REPO_QA_EOF

cat > "src/repo_qa/cli.py" << 'REPO_QA_EOF'
"""
CLI entry point.

This file's only job is orchestration: parse args, call the right modules
in the right order, print results. All actual logic lives in the modules
it imports - keeps this file thin and everything else independently testable.

Usage:
    python -m repo_qa.cli ingest ./my-repo --persist-dir ./chroma_db
    python -m repo_qa.cli query "where is auth handled?" --persist-dir ./chroma_db
"""

import argparse

from repo_qa.config import settings
from repo_qa.ingestion.loader import load_repo
from repo_qa.ingestion.chunker import chunk_documents
from repo_qa.embeddings.provider import get_embedding_function
from repo_qa.vectorstore.chroma_store import build_index, load_index
from repo_qa.retrieval.retriever import retrieve
from repo_qa.generation.answerer import llm_available, generate_answer, format_context


def cmd_ingest(args):
    print(f"Scanning repo: {args.repo_path}")
    docs = load_repo(args.repo_path)
    print(f"Loaded {len(docs)} non-empty documents")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embedding_fn = get_embedding_function(args.persist_dir, force_offline=args.offline)
    print(f"Embedding with: {type(embedding_fn).__name__}")

    build_index(chunks, embedding_fn, args.persist_dir)
    print(f"Index built and persisted to: {args.persist_dir}")


def cmd_query(args):
    embedding_fn = get_embedding_function(args.persist_dir, force_offline=args.offline)
    vectorstore = load_index(embedding_fn, args.persist_dir)

    results = retrieve(vectorstore, args.question, k=args.k)
    if not results:
        print("No relevant chunks found. Did you run `ingest` first?")
        return

    print(f"\nTop {len(results)} retrieved chunks:\n")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"[{i}] {doc.metadata.get('source', 'unknown')}  (distance: {score:.3f}, lower = more similar)")

    if llm_available():
        print("\nGenerating answer with Claude...\n")
        answer = generate_answer(args.question, results)
        print("ANSWER:\n" + answer)
    else:
        print(
            "\n[No ANTHROPIC_API_KEY set - skipping LLM answer step.]\n"
            "Retrieved context that would be sent to the LLM:\n"
        )
        print(format_context(results))


def main():
    parser = argparse.ArgumentParser(prog="repo-qa", description="Read-only Q&A over a local codebase (RAG).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest", help="Index a local repo")
    p_ingest.add_argument("repo_path", help="Path to the local repo to index")
    p_ingest.add_argument("--persist-dir", default=settings.default_persist_dir)
    p_ingest.add_argument("--offline", action="store_true", help="Use offline TF-IDF embeddings")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = subparsers.add_parser("query", help="Ask a question about an indexed repo")
    p_query.add_argument("question", help="Natural language question")
    p_query.add_argument("--persist-dir", default=settings.default_persist_dir)
    p_query.add_argument("-k", type=int, default=None, help=f"Number of chunks to retrieve (default {settings.default_top_k})")
    p_query.add_argument("--offline", action="store_true", help="Use offline TF-IDF embeddings")
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
REPO_QA_EOF

cat > "src/repo_qa/config.py" << 'REPO_QA_EOF'
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
    embedding_model_name: str = _env_str(
        "REPO_QA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
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
REPO_QA_EOF

cat > "src/repo_qa/embeddings/__init__.py" << 'REPO_QA_EOF'

REPO_QA_EOF

cat > "src/repo_qa/embeddings/provider.py" << 'REPO_QA_EOF'
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
            model = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
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
REPO_QA_EOF

cat > "src/repo_qa/generation/__init__.py" << 'REPO_QA_EOF'

REPO_QA_EOF

cat > "src/repo_qa/generation/answerer.py" << 'REPO_QA_EOF'
"""
Answer generation.

Responsibility: turn retrieved chunks + a question into a grounded answer.
This is the only file in the whole pipeline that talks to an LLM.
"""

import os
from langchain_core.documents import Document

from repo_qa.config import settings

ANSWER_PROMPT_TEMPLATE = """You are answering questions about a codebase using ONLY the context below.
If the answer isn't in the context, say so explicitly instead of guessing.
Always mention which file(s) your answer is based on.

Context:
{context}

Question: {question}

Answer:"""


def format_context(results: list[tuple[Document, float]]) -> str:
    """Turn retrieved (chunk, distance) pairs into a labeled context block for the prompt."""
    blocks = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        blocks.append(f"--- {source} (chunk {chunk_idx}, distance {score:.3f}) ---\n{doc.page_content}")
    return "\n\n".join(blocks)


def llm_available() -> bool:
    """Whether we have credentials to actually call an LLM."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def generate_answer(question: str, results: list[tuple[Document, float]]) -> str:
    """Call Claude to generate an answer grounded in the retrieved chunks."""
    from langchain_anthropic import ChatAnthropic

    context = format_context(results)
    llm = ChatAnthropic(model=settings.llm_model_name, max_tokens=settings.llm_max_tokens)
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content
REPO_QA_EOF

cat > "src/repo_qa/ingestion/__init__.py" << 'REPO_QA_EOF'

REPO_QA_EOF

cat > "src/repo_qa/ingestion/chunker.py" << 'REPO_QA_EOF'
"""
Document chunking.

v0 approach: naive character-based splitting via LangChain's
RecursiveCharacterTextSplitter. It's "good enough to work" but can cut a
function in half, since it has no idea what a function is.

This is intentionally isolated in its own module/function (chunk_documents)
so that swapping it for tree-sitter/AST-aware chunking later (see
docs/concepts.md, "Planned Improvements") means changing ONE function, not
hunting through the codebase.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from repo_qa.config import settings


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks, tagged with a per-file chunk index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Tag each chunk with its position within its source file. This makes
    # citations meaningful later ("chunk 2 of auth/login.py") instead of
    # chunks being anonymous blobs of text.
    counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata["source"]
        counters[src] = counters.get(src, 0) + 1
        chunk.metadata["chunk_index"] = counters[src]

    return chunks
REPO_QA_EOF

cat > "src/repo_qa/ingestion/loader.py" << 'REPO_QA_EOF'
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
REPO_QA_EOF

cat > "src/repo_qa/retrieval/__init__.py" << 'REPO_QA_EOF'

REPO_QA_EOF

cat > "src/repo_qa/retrieval/retriever.py" << 'REPO_QA_EOF'
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
REPO_QA_EOF

cat > "src/repo_qa/vectorstore/__init__.py" << 'REPO_QA_EOF'

REPO_QA_EOF

cat > "src/repo_qa/vectorstore/chroma_store.py" << 'REPO_QA_EOF'
"""
Vector store management.

Responsibility: wrap Chroma so the rest of the codebase never talks to
Chroma's API directly. If we ever swap Chroma for Qdrant/Pinecone/pgvector,
this is the only file that should need to change.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from repo_qa.config import settings


def build_index(chunks: list[Document], embedding_fn: Embeddings, persist_dir: str) -> Chroma:
    """Embed and persist a fresh Chroma collection from the given chunks."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=persist_dir,
        collection_name=settings.collection_name,
    )


def load_index(embedding_fn: Embeddings, persist_dir: str) -> Chroma:
    """Load an already-built Chroma collection from disk."""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_fn,
        collection_name=settings.collection_name,
    )
REPO_QA_EOF

cat > "tests/__init__.py" << 'REPO_QA_EOF'

REPO_QA_EOF

cat > "tests/test_chunker.py" << 'REPO_QA_EOF'
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
REPO_QA_EOF

cat > "tests/test_embeddings.py" << 'REPO_QA_EOF'
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
REPO_QA_EOF

echo "Done. Project scaffolded at: $(pwd)"
echo "Next steps:"
echo "  pip install -e ."
echo "  cp .env.example .env   # then add your ANTHROPIC_API_KEY"
echo "  repo-qa ingest sample_repo --persist-dir ./chroma_db"
echo "  repo-qa query \"how long is a session token valid for?\" --persist-dir ./chroma_db"
