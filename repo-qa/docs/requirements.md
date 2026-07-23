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
