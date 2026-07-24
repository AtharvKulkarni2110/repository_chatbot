"""
Answer generation.

Responsibility: turn retrieved chunks + a question into a grounded answer.
This is the only file in the whole pipeline that talks to an LLM.

Provider is swappable via settings.llm_provider ("google" or "anthropic")
specifically so switching to a free-tier provider for learning/experimenting
is a one-line config change, not a rewrite.
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

# Which env var holds the API key, per provider.
_API_KEY_ENV_VARS = {
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),  # either is accepted
    "anthropic": ("ANTHROPIC_API_KEY",),
}


def format_context(results: list[tuple[Document, float]]) -> str:
    """Turn retrieved (chunk, distance) pairs into a labeled context block for the prompt."""
    blocks = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        blocks.append(f"--- {source} (chunk {chunk_idx}, distance {score:.3f}) ---\n{doc.page_content}")
    return "\n\n".join(blocks)


def llm_available() -> bool:
    """Whether we have credentials to actually call an LLM, for the configured provider."""
    env_vars = _API_KEY_ENV_VARS.get(settings.llm_provider, ())
    return any(os.environ.get(var) for var in env_vars)


def _build_llm():
    """Instantiate the configured chat model. Isolated so provider-specific
    imports only happen for whichever provider is actually configured."""
    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.llm_model_name, max_output_tokens=settings.llm_max_tokens)

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=settings.llm_model_name, max_tokens=settings.llm_max_tokens)

    raise ValueError(
        f"Unknown llm_provider '{settings.llm_provider}' - expected 'google' or 'anthropic'. "
        f"Set REPO_QA_LLM_PROVIDER to one of these."
    )


def generate_answer(question: str, results: list[tuple[Document, float]]) -> str:
    """Call the configured LLM to generate an answer grounded in the retrieved chunks."""
    context = format_context(results)
    llm = _build_llm()
    prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content
