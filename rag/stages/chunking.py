"""
Chunking stage: splits parsed markdown into embeddable chunks and enriches
each chunk with its heading ancestry.

This stage owns orchestration — selecting which concrete chunker from
rag.chunkers to run based on config. The chunkers themselves live in
rag/chunkers/; adding a new strategy means dropping a file there and adding
one dispatch line here.
"""
from pathlib import Path

from langchain_core.documents import Document

from rag.config import LOCAL_CHUNKER


def _get_chunker():
    """Select the chunker based on the LOCAL_CHUNKER config flag."""
    if LOCAL_CHUNKER == "docling":
        from rag.chunkers.docling_chunker import DoclingChunker
        return DoclingChunker()
    from rag.chunkers.docling_chunker import DoclingChunker
    return DoclingChunker()


def chunk_and_enrich(
    md_path: Path,
    content_hash: str,
    filename: str,
) -> tuple[list[Document], dict[str, str]]:
    """Route to the configured chunker and return enriched chunks."""
    chunker = _get_chunker()
    splits, parent_headings_text = chunker.chunk(md_path, content_hash, filename)
    print(f"[chunking] Created {len(splits)} chunks, {len(parent_headings_text)} parent heading groups")
    return splits, parent_headings_text
