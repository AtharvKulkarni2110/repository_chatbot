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
