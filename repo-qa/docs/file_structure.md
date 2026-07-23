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
