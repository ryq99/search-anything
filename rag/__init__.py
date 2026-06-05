from pathlib import Path

from rag.indexing import ingest_source, ingest_directory
from rag.retrieval import retrieve, format_retrieval_results, _load_summaries_for_books
from rag.synthesis import ask as _synthesize
from rag.backends.factory import get_backend
from rag.config import BOOKS_DIR


def ask(question: str) -> str:
    """Ask any ML question. Retrieves relevant chunks and synthesizes an answer."""
    backend = get_backend()
    registry = backend.registry.load_all()
    summaries_df = _load_summaries_for_books(registry)
    vs = backend.vectorstore.get_store()
    chunks = retrieve(question, vectorstore=vs, summaries_df=summaries_df)
    retrieval_str = format_retrieval_results(chunks)
    return _synthesize(question, retrieval_str)


def ingest_pdf(pdf_path: Path) -> dict:
    return ingest_source(pdf_path)


def ingest_books(books_dir: Path | None = None) -> list[dict]:
    return ingest_directory(books_dir or BOOKS_DIR)


def start_watcher(books_dir: Path | None = None) -> None:
    """
    Start a blocking watchdog watcher on books_dir.
    Performs a catch-up ingest on startup, then watches for new PDFs live.
    Not suitable for serverless deployment.
    """
    import time
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    from rag.stages.parsing import SUPPORTED_EXTENSIONS

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


__all__ = ["ask", "ingest_pdf", "ingest_books", "start_watcher"]
