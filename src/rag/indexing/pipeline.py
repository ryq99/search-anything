import asyncio
from datetime import datetime, timezone
from pathlib import Path

from rag.ingestion.parsing import parse_document, SUPPORTED_EXTENSIONS
from rag.ingestion.chunking import chunk_and_enrich, save_chunks_jsonl
from rag.ingestion.summarize import summarize_chunks
from rag.backends.factory import get_backend
from rag.core.schemas import BookEntry
from rag.config import BOOKS_DIR, LOCAL_PARSER


def build(parse_result, chunks, backend) -> BookEntry:
    """Embed, store, and register chunks. Returns the BookEntry."""
    backend.vectorstore.store(chunks)
    entry = BookEntry(
        filename=Path(parse_result.source_path).name,
        source_path=parse_result.source_path,
        content_hash=parse_result.content_hash,
        parser=parse_result.parser,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        chunk_count=len(chunks),
    )
    backend.registry.register(entry)
    return entry


def ingest_source(source: Path | str) -> dict:
    """Full pipeline for a single source. Idempotent."""
    backend = get_backend()
    source = Path(source)

    print(f"[pipeline] Parsing {source.name}...")
    parse_result = parse_document(source)

    if backend.registry.is_ingested(parse_result.content_hash, parse_result.parser):
        print(f"[pipeline] Already ingested: {source.name} ({parse_result.parser}, hash={parse_result.content_hash[:10]}...)")
        return backend.registry.get(parse_result.content_hash, parse_result.parser)

    print("[pipeline] Chunking and enriching...")
    chunks = chunk_and_enrich(parse_result)

    print(f"[pipeline] Summarizing {len(chunks)} chunks...")
    asyncio.run(summarize_chunks(chunks, backend.summary_llm))

    if parse_result.doc_dir is not None:
        save_chunks_jsonl(chunks, parse_result.doc_dir)

    print(f"[pipeline] Embedding and storing {len(chunks)} chunks...")
    entry = build(parse_result, chunks, backend)

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
        already = any(
            e.get("filename") == path.name and e.get("parser") == LOCAL_PARSER
            for e in all_entries.values()
        )
        if already:
            print(f"[pipeline] Skipping (already ingested with {LOCAL_PARSER}): {path.name}")
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
