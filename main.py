import argparse
from pathlib import Path

from rag.retrieval import ask
from rag.indexing import ingest_source, ingest_directory, start_watcher
from rag.config import BOOKS_DIR


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
            results = [ingest_source(p) for p in args.paths]
        else:
            results = ingest_directory(BOOKS_DIR)
        if results:
            print(f"\nIngested {len(results)} new file(s):")
            for r in results:
                print(f"  - {r['filename']} ({r['chunk_count']} chunks)")
        else:
            print("No new files to ingest.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
