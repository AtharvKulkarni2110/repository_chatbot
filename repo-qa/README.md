# Repository-Aware AI Code Assistant

## Repository-aware Question Answering using Retrieval-Augmented Generation (RAG)

A modular AI assistant that indexes local source code repositories and answers natural language questions using semantic search, AST-aware code chunking, vector embeddings, and Large Language Models.

![Python](https://img.shields.io/badge/Python-3.10+-2563EB?style=for-the-badge\&logo=python\&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1D4ED8?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-3B82F6?style=for-the-badge)
![Tree-sitter](https://img.shields.io/badge/Tree--sitter-AST-60A5FA?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Gemini-LLM-2563EB?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-1E40AF?style=for-the-badge)

---

# Overview

Understanding an unfamiliar codebase can take hours. Developers often spend significant time searching across multiple files before finding the implementation of a feature, API, or business logic.

**Repository-Aware AI Code Assistant** addresses this problem by transforming a local repository into a searchable knowledge base.

Instead of relying on keyword matching, the system:

* Parses source code
* Extracts meaningful code units
* Generates semantic vector embeddings
* Stores them inside a vector database
* Retrieves the most relevant code
* Produces grounded answers using an LLM with source citations

The project is designed with a modular architecture so that every stage of the RAG pipeline can be independently developed, tested, and replaced.

---

# Demo

## Application Demo

> **Demo GIF / Video**

```
docs/demo/demo.gif
```

*(Add a screen recording of asking questions and receiving answers.)*

---

## User Interface

> **Screenshot Placeholder**

```
docs/images/chat-interface.png
```

---

## Retrieval Visualization

> **Screenshot Placeholder**

```
docs/images/retrieval.png
```

---

# Features

* AST-aware code chunking using Tree-sitter
* Semantic retrieval using vector embeddings
* Source-aware answers with file citations
* Modular Retrieval-Augmented Generation pipeline
* Persistent ChromaDB vector store
* Offline TF-IDF fallback when embedding models are unavailable
* Pluggable embedding and LLM providers
* Repository indexing through a simple CLI
* Chat interface for querying indexed repositories
* Clean separation between ingestion, retrieval, and generation

---

# Architecture

```
                    Repository

                         │
                         ▼

                 Repository Loader

                         │
                         ▼

                 AST-aware Chunker

                         │
                         ▼

                 Embedding Generator

                         │
                         ▼

                  Chroma Vector Store

                         │
             ─────────────────────────
                         │
                   User Question
                         │
                         ▼
                  Query Embedding
                         │
                         ▼
                 Similarity Search
                         │
                         ▼
              Top-K Relevant Chunks
                         │
                         ▼
                Prompt Construction
                         │
                         ▼
                    Gemini LLM
                         │
                         ▼
              Grounded Final Response
```

---

# Project Structure

```
repo-qa
│
├── src
│   └── repo_qa
│       ├── cli.py
│       ├── config.py
│       │
│       ├── ingestion
│       │   ├── loader.py
│       │   └── chunker.py
│       │
│       ├── embeddings
│       │   └── provider.py
│       │
│       ├── vectorstore
│       │   └── chroma_store.py
│       │
│       ├── retrieval
│       │   └── retriever.py
│       │
│       └── generation
│           └── answerer.py
│
├── docs
├── tests
├── README.md
└── requirements.txt
```

---

# Technology Stack

| Category        | Technology            |
| --------------- | --------------------- |
| Language        | Python                |
| Framework       | LangChain             |
| Vector Database | ChromaDB              |
| Parser          | Tree-sitter           |
| Embeddings      | Hugging Face / TF-IDF |
| LLM             | Gemini                |
| Frontend        | React *(planned)*     |
| Backend API     | FastAPI *(planned)*   |
| Testing         | Pytest                |

---

# Retrieval Pipeline

```
Repository

↓

Load Files

↓

Parse Files

↓

Extract Functions

↓

Generate Embeddings

↓

Store Vectors

──────────────────────────────

User Question

↓

Generate Query Embedding

↓

Similarity Search

↓

Top-K Chunks

↓

LLM

↓

Answer + Citations
```

---

# Why AST-aware Chunking?

Traditional RAG implementations split files into fixed character windows.

```
Character Splitter

------------------------
def login():
    ...
------------------------
authenticate(...)
------------------------
```

This often breaks functions into multiple chunks.

Instead, this project uses **Tree-sitter** to split repositories at **function and method boundaries**.

```
Function login()

↓

One Chunk


Function authenticate()

↓

One Chunk
```

Benefits include:

* Preserves complete logical units
* Improves retrieval quality
* Reduces irrelevant context
* Produces cleaner prompts for the LLM

---

# Design Principles

The project follows a modular architecture where every component has a single responsibility.

| Module       | Responsibility                          |
| ------------ | --------------------------------------- |
| Loader       | Reads repository files                  |
| Chunker      | Splits documents into meaningful chunks |
| Embeddings   | Converts chunks into vectors            |
| Vector Store | Stores and retrieves vectors            |
| Retriever    | Performs similarity search              |
| Generator    | Produces grounded answers               |
| CLI          | Orchestrates the entire pipeline        |

This design makes every module independently testable and replaceable.

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/repo-qa.git

cd repo-qa
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -e .
```

---

# Environment Variables

Create a `.env` file.

```
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY

HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

---

# Usage

## Index a Repository

```bash
repo-qa ingest ./sample_repo
```

---

## Ask Questions

```bash
repo-qa query "Where is authentication handled?"
```

Example questions

```
Where is the JWT token generated?

How is the database connection initialized?

Where are API routes defined?

Explain the login flow.

Where is password hashing implemented?
```

---

# Example Output

```
Question

Where is authentication handled?

────────────────────────────

Answer

Authentication is implemented inside
auth/login.py.

The login() function validates user
credentials before generating a JWT.

Sources

auth/login.py
auth/token.py
```

---

# Testing

Run the test suite

```bash
pytest tests -v
```

The project includes unit tests for

* AST chunking
* Character chunking
* Embedding generation
* Metadata generation
* Retrieval pipeline

---

# Current Limitations

* Cross-file reasoning is limited
* No repository editing capabilities
* No agent workflow
* No incremental indexing
* Frontend currently under development

---

# Future Roadmap

### Backend

* FastAPI REST API
* Streaming responses
* Incremental indexing
* Metadata filtering
* Hybrid retrieval

### Frontend

* React chatbot interface
* Repository selection
* Dark/Light mode
* Citation highlighting
* Retrieved chunk viewer
* Conversation history

### AI Improvements

* Parent class metadata
* Cross-file reasoning
* Hybrid BM25 + Vector Search
* Query rewriting
* Re-ranking models
* Multi-agent architecture

---

# Performance Goals

* Repository indexing under one minute for medium-sized repositories
* Function-level semantic retrieval
* Source-grounded responses
* Offline retrieval support using TF-IDF
* Fully modular and replaceable architecture

---

# Contributing

Contributions are welcome.

If you have ideas for improving retrieval quality, supporting additional programming languages, or enhancing the user experience, feel free to open an issue or submit a pull request.

---

# License

This project is released under the **MIT License**.

---

Built as part of a modular AI Software Engineering project focused on Retrieval-Augmented Generation, semantic code search, and repository understanding.
