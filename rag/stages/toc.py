import re
from pathlib import Path

import pandas as pd

from rag.core.interfaces import LLM
from rag.config import DATA_DIR

_TOC_START = re.compile(
    r'^#{0,6}\s*(table\s+of\s+contents|contents)\s*$',
    re.IGNORECASE | re.MULTILINE,
)
_TOC_WINDOW = 50_000


def _extract_toc_section(markdown_text: str) -> str:
    start_match = _TOC_START.search(markdown_text)
    if not start_match:
        return markdown_text[:_TOC_WINDOW]
    start = start_match.start()
    return markdown_text[start:start + _TOC_WINDOW]


def extract_toc(markdown_text: str, stem: str, llm: LLM) -> pd.DataFrame:
    """Extract hierarchical TOC from markdown using the provided LLM. Saves *_toc.csv."""
    toc_excerpt = _extract_toc_section(markdown_text)

    system = (
        "Extract table-of-contents hierarchy from markdown. "
        "Each hierarchy level must include BOTH the section number and the section title. "
        "Format each path as: <number> <title> => <number> <title> => ... "
        "Use any depth. Output only the paths. "
        "Return an empty string if no table of contents is present."
    )
    user = (
        "Extract the table of contents from the markdown below.\n"
        "Include the section number and section title at every level.\n\n"
        "Example:\n"
        "1 Introduction => 1.2 Random Variables => 1.2.2 Distributions\n\n"
        f"{toc_excerpt}"
    )

    raw = llm.complete(system=system, user=user, max_tokens=4096)

    toc_value = [
        row.strip().split(" => ")
        for row in raw.split("\n")
        if row.strip()
    ]

    if not toc_value:
        return pd.DataFrame()

    max_depth = max(len(row) for row in toc_value)
    toc_value = [row + [""] * (max_depth - len(row)) for row in toc_value]

    toc = pd.DataFrame(
        toc_value,
        columns=[f"Level {i + 1}" for i in range(max_depth)],
    ).fillna("")

    out_file = DATA_DIR / f"{stem}_toc.csv"
    toc.to_csv(out_file, index=False)
    print(f"[toc] Saved TOC: {out_file}")
    return toc
