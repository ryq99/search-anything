import asyncio
from datetime import datetime, timezone
from pathlib import Path

from rag.parsers.router import parse_to_markdown, SUPPORTED_EXTENSIONS
from rag.stages import toc as toc_stage
from rag.stages import chunking as chunking_stage
from rag.stages import summarize as summarize_stage
from rag.backends.factory import get_backend
from rag.core.models import BookEntry
from rag.config import DATA_DIR, BOOKS_DIR


def _stem(source_path: str) -> str:
    return Path(source_path).stem.replace(" ", "_").replace("-", "_").lower()


def ingest_source(source: Path | str) -> dict:
    """Full ingestion pipeline for a single source. Idempotent."""
    backend = get_backend()
    source = Path(source)

    print(f"[pipeline] Parsing {source.name}...")
    parse_result = parse_to_markdown(source)

    if backend.registry.is_ingested(parse_result.content_hash):
        print(f"[pipeline] Already ingested: {source.name} (hash={parse_result.content_hash[:10]}...)")
        return backend.registry.get(parse_result.content_hash)

    stem = _stem(parse_result.source_path)
    md_path = DATA_DIR / f"{stem}_converted.md"

    print("[pipeline] Extracting TOC...")
    toc_df = toc_stage.extract_toc(parse_result.content, stem, backend.llm)

    print("[pipeline] Chunking and enriching...")
    chunks, parent_headings_text = chunking_stage.chunk_and_enrich(
        md_path, toc_df, parse_result.content_hash, source.name
    )

    print(f"[pipeline] Summarizing {len(parent_headings_text)} parent heading groups...")
    summaries = asyncio.run(summarize_stage.summarize_all(parent_headings_text, backend.llm))
    summary_path = summarize_stage.save_summaries_csv(summaries, stem)

    print(f"[pipeline] Embedding and storing {len(chunks)} chunks...")
    backend.vectorstore.store(chunks)

    entry = BookEntry(
        filename=source.name,
        source_path=str(source.absolute()),
        content_hash=parse_result.content_hash,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        chunk_count=len(chunks),
        toc_artifact_path=str(DATA_DIR / f"{stem}_toc.csv"),
        summary_artifact_path=str(summary_path),
    )
    backend.registry.register(entry)

    print(f"[pipeline] Done: {source.name} ({len(chunks)} chunks)")
    return entry.to_dict()


def ingest_directory(directory: Path | None = None) -> list[dict]:
    """Ingest all supported files in a directory, skipping already-ingested ones."""
    directory = directory or BOOKS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    backend = get_backend()
    all_entries = backend.registry.load_all().get("books", {})

    results = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        already = any(e.get("filename") == path.name for e in all_entries.values())
        if already:
            print(f"[pipeline] Skipping (already ingested): {path.name}")
            continue
        results.append(ingest_source(path))
    return results
