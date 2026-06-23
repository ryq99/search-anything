import asyncio
from pathlib import Path

from rag.ingestion.parsing import parse_document
from rag.ingestion.chunking import chunk_and_enrich, save_chunks_jsonl
from rag.ingestion.summarize import summarize_chunks
from rag.core.schemas import ParseResult, Chunk


def run(source: Path | str, summary_llm, parse_result: ParseResult | None = None) -> tuple[ParseResult, list[Chunk]]:
    """Parse, chunk, and summarize a single source. Returns (ParseResult, list[Chunk])."""
    if parse_result is None:
        parse_result = parse_document(Path(source))
    chunks = chunk_and_enrich(parse_result)
    asyncio.run(summarize_chunks(chunks, summary_llm))
    if parse_result.doc_dir is not None:
        save_chunks_jsonl(chunks, parse_result.doc_dir)
    return parse_result, chunks
