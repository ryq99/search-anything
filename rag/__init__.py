from pathlib import Path

from rag.ingestion import ingest_books, ingest_pdf, load_processed_books
from rag.retrieval import (
    _get_vectorstore,
    _load_summaries_for_books,
    retrieve,
    format_retrieval_results,
)
from rag.synthesis import ask as _synthesize
from rag.config import BOOKS_DIR


def ask(question: str) -> str:
    """
    Ask any machine learning question. Returns a structured answer.

    Retrieves relevant chunks from the ingested book vector store,
    enriches them with pre-computed section summaries, and synthesizes
    an answer using Claude.

    Lambda-compatible: no global state held between calls.
    """
    registry = load_processed_books()
    summaries_df = _load_summaries_for_books(registry)
    vs = _get_vectorstore()
    chunks = retrieve(question, vectorstore=vs, summaries_df=summaries_df)
    retrieval_str = format_retrieval_results(chunks)
    return _synthesize(question, retrieval_str)


def start_watcher(books_dir: Path | None = None) -> None:
    """
    Start a blocking watchdog watcher on books_dir.
    Performs a catch-up ingest on startup, then watches for new PDFs live.
    Not Lambda-compatible — use in main.py only.
    """
    import time
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent

    books_dir = books_dir or BOOKS_DIR
    books_dir.mkdir(parents=True, exist_ok=True)

    class _PDFHandler(FileSystemEventHandler):
        def on_created(self, event: FileCreatedEvent) -> None:
            if not event.is_directory and Path(event.src_path).suffix.lower() == ".pdf":
                print(f"[watcher] New PDF detected: {event.src_path}")
                try:
                    result = ingest_pdf(Path(event.src_path))
                    print(f"[watcher] Ingested: {result['filename']}")
                except Exception as e:
                    print(f"[watcher] Ingestion failed for {event.src_path}: {e}")

    print(f"[watcher] Scanning {books_dir} for unprocessed PDFs...")
    ingest_books(books_dir)

    observer = Observer()
    observer.schedule(_PDFHandler(), str(books_dir), recursive=False)
    observer.start()
    print(f"[watcher] Watching {books_dir} for new PDFs. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


__all__ = ["ask", "ingest_books", "ingest_pdf", "start_watcher"]
