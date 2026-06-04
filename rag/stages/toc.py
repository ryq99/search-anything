import re
from pathlib import Path

import pandas as pd

from rag.core.interfaces import LLM
from rag.config import DATA_DIR

# Headings that signal the start of a TOC section
_TOC_START = re.compile(
    r'^#{0,6}\s*(table\s+of\s+contents|contents|toc)\s*$',
    re.IGNORECASE | re.MULTILINE,
)

# A heading that signals the TOC has ended (first chapter/section/preface after TOC)
_SECTION_HEADING = re.compile(
    r'^#{1,3}\s+\S',
    re.MULTILINE,
)

# Page-number suffixes common in parsed TOC lines: "Chapter 1 ........ 12" or "Chapter 1 12"
_PAGE_NUMBER_SUFFIX = re.compile(r'[\s.·\-]{2,}\d+\s*$')

_TOC_MAX_WINDOW = 60_000   # hard cap if section end can't be found
_TOC_FALLBACK_WINDOW = 50_000  # window when no TOC heading found


def _extract_toc_section(markdown_text: str) -> str:
    """
    Return the slice of markdown most likely to contain the TOC.

    Strategy:
    1. Find the TOC heading ("Table of Contents", "Contents", "TOC").
    2. Find the *next* major heading after it (signals TOC has ended).
    3. Return the slice between them, capped at _TOC_MAX_WINDOW chars.
    4. Fall back to the first _TOC_FALLBACK_WINDOW chars if no TOC heading found.
    """
    start_match = _TOC_START.search(markdown_text)
    if not start_match:
        return markdown_text[:_TOC_FALLBACK_WINDOW]

    toc_start = start_match.end()
    cap = min(toc_start + _TOC_MAX_WINDOW, len(markdown_text))
    search_region = markdown_text[toc_start:cap]

    # Find the first major heading that follows the TOC start
    end_match = _SECTION_HEADING.search(search_region)
    toc_end = toc_start + end_match.start() if end_match else cap

    return markdown_text[start_match.start():toc_end]


def _strip_page_numbers(line: str) -> str:
    """Remove trailing page numbers like '....... 42' or '  42'."""
    return _PAGE_NUMBER_SUFFIX.sub("", line).strip()


def extract_toc(markdown_text: str, stem: str, llm: LLM) -> pd.DataFrame:
    """Extract hierarchical TOC from markdown using the provided LLM. Saves *_toc.csv."""
    toc_excerpt = _extract_toc_section(markdown_text)

    system = (
        "You extract structured table-of-contents hierarchies from book markdown.\n"
        "Rules:\n"
        "- Output one TOC entry per line as a ' => '-delimited path, deepest entry last.\n"
        "- Each segment must be '<number_or_letter> <title>' (e.g. '1 Introduction', 'A Appendix', 'I Preface').\n"
        "- Omit page numbers. Omit blank or separator lines.\n"
        "- If the markdown contains NO table of contents, output exactly: NO_TOC\n"
        "- Do not explain or add commentary.\n\n"
        "Example output:\n"
        "1 Introduction\n"
        "1 Introduction => 1.1 Motivation\n"
        "1 Introduction => 1.2 Overview\n"
        "2 Background => 2.1 Related Work"
    )
    user = (
        "Extract the complete table of contents hierarchy from the markdown below.\n\n"
        f"{toc_excerpt}"
    )

    raw = llm.complete(system=system, user=user, max_tokens=4096).strip()

    if raw == "NO_TOC" or not raw:
        print(f"[toc] No TOC found in '{stem}'")
        return pd.DataFrame()

    rows: list[list[str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line == "NO_TOC":
            continue
        segments = [_strip_page_numbers(seg) for seg in line.split(" => ")]
        segments = [s for s in segments if s]
        if segments:
            rows.append(segments)

    if not rows:
        return pd.DataFrame()

    max_depth = max(len(row) for row in rows)
    padded = [row + [""] * (max_depth - len(row)) for row in rows]

    toc = pd.DataFrame(
        padded,
        columns=[f"Level {i + 1}" for i in range(max_depth)],
    ).fillna("")

    out_file = DATA_DIR / f"{stem}_toc.csv"
    toc.to_csv(out_file, index=False)
    print(f"[toc] Saved TOC: {out_file} ({len(toc)} entries, depth {max_depth})")
    return toc
