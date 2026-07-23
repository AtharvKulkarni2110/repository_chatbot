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
