"""
AST-aware document chunking (v1 — replaces v0's character-based splitting).

Chunks are drawn at function/method boundaries via tree-sitter, so a chunk
is always a complete, meaningful unit of code — matching how a human
actually reads code (as callable units), not an arbitrary character window
that can cut a function in half.

Strategy per file:
1. If the file's language has a tree-sitter grammar available, parse it and
   extract every function/method definition as its own chunk.
2. If a single extracted unit is unusually large (e.g. a huge function),
   sub-split it with the old character-based splitter so it doesn't exceed
   what the embedding model can meaningfully encode.
3. If the language isn't supported, or the file has no functions/methods at
   all (e.g. a config file, a dataclass with no methods, a markdown doc),
   fall back to the old character-based splitter for that whole file.

This means ingestion never hard-fails on an unsupported file type — it just
degrades to v0 behavior for that one file.

Known limitation (intentional, for now): a method chunk doesn't carry its
enclosing class name in its text or metadata yet, so "class Foo: def
bar()" shows up as a chunk for just `bar`, without "Foo" for context. Fine
for most single-purpose methods; a planned follow-up is to attach
`parent_class` metadata by tracking ancestry while walking the tree.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from repo_qa.config import settings

# Maps file extension -> (tree-sitter language name, node types treated as
# a "chunkable unit"). Deliberately using the smallest meaningful callable
# unit (function/method), not class/module, so a chunk is never bigger than
# it needs to be. A class with no methods (e.g. a plain dataclass) simply
# won't match anything here and falls through to the character splitter.
LANGUAGE_MAP = {
    ".py": ("python", {"function_definition"}),
    ".js": ("javascript", {"function_declaration", "method_definition"}),
    ".jsx": ("javascript", {"function_declaration", "method_definition"}),
    ".ts": ("typescript", {"function_declaration", "method_definition"}),
    ".tsx": ("tsx", {"function_declaration", "method_definition"}),
    ".java": ("java", {"method_declaration"}),
    ".go": ("go", {"function_declaration", "method_declaration"}),
    ".rb": ("ruby", {"method"}),
    ".c": ("c", {"function_definition"}),
    ".cpp": ("cpp", {"function_definition"}),
    ".rs": ("rust", {"function_item"}),
}

# If an extracted AST chunk is bigger than this, sub-split it further.
# Generous multiple of chunk_size: most functions are fine as one chunk;
# this only kicks in for genuinely oversized functions, so we don't
# silently exceed what the embedding model can encode in one pass.
MAX_AST_CHUNK_CHARS_MULTIPLIER = 3

_parser_cache: dict[str, object] = {}


def _get_parser(language_name: str):
    """Load and cache a tree-sitter parser for the given language."""
    if language_name not in _parser_cache:
        from tree_sitter_language_pack import get_parser
        _parser_cache[language_name] = get_parser(language_name)
    return _parser_cache[language_name]


def _language_for_source(source_path: str):
    """Look up (language_name, unit_types) for a file path, or None if unsupported."""
    for ext, config in LANGUAGE_MAP.items():
        if source_path.endswith(ext):
            return config
    return None


def _extract_ast_chunks(doc: Document, language_name: str, unit_types: set) -> list[Document] | None:
    """
    Try to split one document into function/method-level chunks.
    Returns None if the language can't be parsed at all (missing grammar,
    parser error) so the caller knows to fall back - as opposed to an empty
    list, which means "parsed fine, just no matching units in this file".
    """
    try:
        parser = _get_parser(language_name)
    except Exception:
        return None

    try:
        tree = parser.parse(doc.page_content.encode("utf-8"))
    except Exception:
        return None

    chunks: list[Document] = []

    def walk(node):
        if node.type in unit_types:
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf-8", errors="ignore") if name_node else None
            chunks.append(Document(
                page_content=node.text.decode("utf-8", errors="ignore"),
                metadata={
                    **doc.metadata,
                    "node_type": node.type,
                    "function_name": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                },
            ))
            return  # don't descend further - keeps the outer unit as one chunk, not split into nested pieces
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks


def _character_split(doc: Document) -> list[Document]:
    """The v0 fallback splitter, used for unsupported languages and oversized chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents([doc])


def _split_oversized_chunk(chunk: Document) -> list[Document]:
    """If an AST-extracted chunk is unusually large, sub-split it further."""
    max_chars = settings.chunk_size * MAX_AST_CHUNK_CHARS_MULTIPLIER
    if len(chunk.page_content) <= max_chars:
        return [chunk]
    return _character_split(chunk)


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents into chunks, preferring function/method-boundary chunks
    via tree-sitter, falling back to character-based splitting per-file
    when the language is unsupported or the file has no matching units.
    """
    all_chunks: list[Document] = []

    for doc in docs:
        source = doc.metadata.get("source", "")
        lang_config = _language_for_source(source)

        ast_chunks = None
        if lang_config:
            language_name, unit_types = lang_config
            ast_chunks = _extract_ast_chunks(doc, language_name, unit_types)

        if ast_chunks:  # non-empty list of real function/method chunks
            for chunk in ast_chunks:
                all_chunks.extend(_split_oversized_chunk(chunk))
        else:  # ast_chunks is None (unsupported language) or [] (no functions found)
            all_chunks.extend(_character_split(doc))

    # Tag each chunk with its position within its source file, for citations.
    counters: dict[str, int] = {}
    for chunk in all_chunks:
        src = chunk.metadata["source"]
        counters[src] = counters.get(src, 0) + 1
        chunk.metadata["chunk_index"] = counters[src]

    return all_chunks
