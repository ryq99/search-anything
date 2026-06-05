"""
Chunking step: splits a parsed document into embeddable chunks and enriches
each with heading ancestry.

Orchestration only — selects the configured chunker from rag.chunkers and
routes the ParseResult to it. Adding a strategy = drop a file in chunkers/
and add one dispatch line in _get_chunker().
"""
from rag.core.schemas import Chunk, ParseResult
from rag.config import LOCAL_CHUNKER


def _get_chunker():
    if LOCAL_CHUNKER == "docling":
        from rag.chunkers.docling_chunker import DoclingChunker
        return DoclingChunker()
    from rag.chunkers.docling_chunker import DoclingChunker
    return DoclingChunker()


def chunk_and_enrich(parse_result: ParseResult) -> tuple[list[Chunk], dict[str, str]]:
    """Route to the configured chunker and return enriched Chunk objects."""
    chunker = _get_chunker()
    chunks, parent_headings_text = chunker.chunk(parse_result)
    print(f"[chunking] Created {len(chunks)} chunks, {len(parent_headings_text)} parent heading groups")
    return chunks, parent_headings_text
