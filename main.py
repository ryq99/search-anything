import argparse
from pathlib import Path

from rag import ask, ingest_books, start_watcher
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

    subparsers.add_parser("watch", help="Watch books/ folder and auto-ingest new PDFs")

    ask_parser = subparsers.add_parser("ask", help="Ask a machine learning question")
    ask_parser.add_argument("question", type=str, help="Your question")

    subparsers.add_parser("ingest", help="One-shot ingest all PDFs in books/")

    args = parser.parse_args()

    if args.command == "watch":
        start_watcher(BOOKS_DIR)

    elif args.command == "ask":
        answer = ask(args.question)
        print(answer)

    elif args.command == "ingest":
        results = ingest_books(BOOKS_DIR)
        if results:
            print(f"\nIngested {len(results)} new book(s):")
            for r in results:
                print(f"  - {r['filename']} ({r['chunk_count']} chunks)")
        else:
            print("No new books to ingest.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
