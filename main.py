"""
Dev entrypoint shim. Lets `python main.py ...` work without installing.

The real CLI lives in the package at src/rag/cli.py and is also exposed as the
`search-anything` console script (see pyproject.toml [project.scripts]).
"""
import sys
from pathlib import Path

# Make src/ importable when running from a source checkout without `pip install`
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.cli import main

if __name__ == "__main__":
    main()
