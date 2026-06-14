"""
Chunking step: splits a parsed document into embeddable chunks and enriches
each with heading ancestry.

Orchestration only — selects the configured chunker from rag.chunkers and
routes the ParseResult to it. Adding a strategy = drop a file in chunkers/
and add one dispatch line in _get_chunker().
"""
import dataclasses
import json
from pathlib import Path

from rag.core.schemas import Chunk, ParseResult
from rag.config import LOCAL_CHUNKER


def _get_chunker():
    if LOCAL_CHUNKER == "docling":
        from rag.chunkers.docling_chunker import DoclingChunker
        return DoclingChunker()
    from rag.chunkers.liteparse_chunker import LiteParseChunker
    return LiteParseChunker()


def save_chunks_jsonl(chunks: list[Chunk], doc_dir: Path) -> Path:
    chunks_dir = doc_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_file = chunks_dir / "chunks.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(dataclasses.asdict(chunk)) + "\n")
    print(f"[chunking] Saved {len(chunks)} chunks: {out_file}")
    return out_file


def chunk_and_enrich(parse_result: ParseResult) -> list[Chunk]:
    """Route to the configured chunker and return enriched Chunk objects."""
    chunker = _get_chunker()
    chunks = chunker.chunk(parse_result)
    print(f"[chunking] Created {len(chunks)} chunks")
    return chunks
