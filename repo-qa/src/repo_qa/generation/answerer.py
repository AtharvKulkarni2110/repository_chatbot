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
