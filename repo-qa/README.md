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