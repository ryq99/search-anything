import asyncio
from datetime import datetime, timezone
from pathlib import Path

from rag.steps.parsing import parse_document, SUPPORTED_EXTENSIONS
from rag.steps import chunking as chunking_step
from rag.steps import summarize as summarize_step
from rag.backends.factory import get_backend
from rag.core.schemas import BookEntry
from rag.config import BOOKS_DIR


def ingest_source(source: Path | str) -> dict:
    """Full ingestion pipeline for a single source. Idempotent."""
    backend = get_backend()
    source = Path(source)

    print(f"[pipeline] Parsing {source.name}...")
    parse_result = parse_document(source)

    if backend.registry.is_ingested(parse_result.content_hash, parse_result.parser):
        print(f"[pipeline] Already ingested: {source.name} ({parse_result.parser}, hash={parse_result.content_hash[:10]}...)")
        return backend.registry.get(parse_result.content_hash, parse_result.parser)

    print("[pipeline] Chunking and enriching...")
    chunks, parent_headings_text = chunking_step.chunk_and_enrich(parse_result)

    print(f"[pipeline] Summarizing {len(parent_headings_text)} parent heading groups...")
    summaries = asyncio.run(summarize_step.summarize_all(parent_headings_text, backend.llm))
    summary_path = summarize_step.save_summaries_csv(summaries, parse_result.doc_dir)

    print(f"[pipeline] Embedding and storing {len(chunks)} chunks...")
    backend.vectorstore.store(chunks)

    entry = BookEntry(
        filename=source.name,
        source_path=str(source.absolute()),
        content_hash=parse_result.content_hash,
        parser=parse_result.parser,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        chunk_count=len(chunks),
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


def start_watcher(books_dir: Path | None = None) -> None:
    """
    Start a blocking watchdog watcher on books_dir.
    Performs a catch-up ingest on startup, then watches for new files live.
    Not suitable for serverless deployment.
    """
    import time
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent

    books_dir = books_dir or BOOKS_DIR
    books_dir.mkdir(parents=True, exist_ok=True)

    class _Handler(FileSystemEventHandler):
        def on_created(self, event: FileCreatedEvent) -> None:
            path = Path(event.src_path)
            if not event.is_directory and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                print(f"[watcher] New file detected: {path.name}")
                try:
                    result = ingest_source(path)
                    print(f"[watcher] Ingested: {result['filename']}")
                except Exception as e:
                    print(f"[watcher] Ingestion failed for {path.name}: {e}")

    print(f"[watcher] Scanning {books_dir} for unprocessed files...")
    ingest_directory(books_dir)

    observer = Observer()
    observer.schedule(_Handler(), str(books_dir), recursive=False)
    observer.start()
    print(f"[watcher] Watching {books_dir}. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
