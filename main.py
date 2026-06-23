import argparse
import time
from pathlib import Path

from rag.ingestion.parsing import parse_document, SUPPORTED_EXTENSIONS
from rag.ingestion.pipeline import run as ingest_run
from rag.indexing.store import build
from rag.inference.retrieval import ask
from rag.backends.factory import get_backend
from rag.config import BOOKS_DIR, LOCAL_PARSER


def index_source(source: Path | str) -> dict:
    """Run ingestion then indexing for a single source. Idempotent."""
    backend = get_backend()
    source = Path(source)

    print(f"[pipeline] Parsing {source.name}...")
    parse_result = parse_document(source)

    if backend.registry.is_ingested(parse_result.content_hash, parse_result.parser):
        print(f"[pipeline] Already indexed: {source.name} ({parse_result.parser}, hash={parse_result.content_hash[:10]}...)")
        return backend.registry.get(parse_result.content_hash, parse_result.parser)

    print("[pipeline] Chunking, enriching, and summarizing...")
    parse_result, chunks = ingest_run(source, backend.summary_llm, parse_result=parse_result)

    print(f"[pipeline] Embedding and storing {len(chunks)} chunks...")
    entry = build(parse_result, chunks, backend)

    print(f"[pipeline] Done: {source.name} ({len(chunks)} chunks)")
    return entry.to_dict()


def ingest_directory(directory: Path | None = None) -> list[dict]:
    """Index all supported files in a directory, skipping already-indexed ones."""
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
            print(f"[pipeline] Skipping (already indexed with {LOCAL_PARSER}): {path.name}")
            continue
        results.append(index_source(path))
    return results


def start_watcher(books_dir: Path | None = None) -> None:
    """
    Start a blocking watchdog watcher on books_dir.
    Performs a catch-up ingest on startup, then watches for new files live.
    Not suitable for serverless deployment.
    """
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
                    result = index_source(path)
                    print(f"[watcher] Indexed: {result['filename']}")
                except Exception as e:
                    print(f"[watcher] Indexing failed for {path.name}: {e}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Search Anything — ML Book RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py ingest\n"
            '  python main.py ask "What is gradient descent?"\n'
            "  python main.py watch\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("watch", help="Watch books/ folder and auto-ingest new files")

    ask_parser = subparsers.add_parser("ask", help="Ask a machine learning question")
    ask_parser.add_argument("question", type=str, help="Your question")

    ingest_parser = subparsers.add_parser("ingest", help="One-shot ingest all files in books/")
    ingest_parser.add_argument(
        "--paths", nargs="+", type=Path, metavar="FILE",
        help="Specific file(s) to ingest (default: all supported files in books/)",
    )

    args = parser.parse_args()

    if args.command == "watch":
        start_watcher(BOOKS_DIR)

    elif args.command == "ask":
        print(ask(args.question))

    elif args.command == "ingest":
        if args.paths:
            results = [index_source(p) for p in args.paths]
        else:
            results = ingest_directory(BOOKS_DIR)
        if results:
            print(f"\nIndexed {len(results)} new file(s):")
            for r in results:
                print(f"  - {r['filename']} ({r['chunk_count']} chunks)")
        else:
            print("No new files to index.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
