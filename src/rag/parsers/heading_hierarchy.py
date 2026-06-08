import re

# Font name substrings that identify heading-style text in liteparse output.
_HEADING_FONT_MARKERS = ("Demi", "Bold")


def annotate_markdown_with_headings(markdown: str, pages: list[dict]) -> str:
    """
    Re-annotate liteparse markdown with # heading markers inferred from font metadata.

    LiteParse outputs layout-preserving markdown without structural heading markers.
    This function detects headings via font name (substrings in _HEADING_FONT_MARKERS),
    then inserts # markers at the correct depth (derived from section-number prefix)
    so that downstream markdown-based chunkers can split on structural boundaries.

    The heading text detected from pages is matched against lines in the markdown
    by substring, preserving the original body text verbatim.
    """
    headings = _extract_headings_from_pages(pages)
    if not headings:
        return markdown

    lines = markdown.splitlines()
    annotated: list[str] = []
    for line in lines:
        stripped = line.strip()
        matched = next((h for h in headings if stripped == h or stripped.startswith(h)), None)
        if matched:
            level = _heading_level(matched)
            annotated.append(f"{'#' * level} {stripped}")
        else:
            annotated.append(line)
    return "\n".join(annotated)


def _extract_headings_from_pages(pages: list[dict]) -> list[str]:
    """
    Extract heading strings from liteparse page font metadata.

    Consecutive heading-font items are joined; parts starting with '.' are
    appended without a space to reconstruct split section numbers like
    '1.2' + '.1' → '1.2.1'.
    """
    headings: list[str] = []
    pending: list[str] = []

    for page in pages:
        for item in page.get("text_items", []):
            text = item.get("text", "").strip()
            if not text:
                continue
            if any(m in item.get("font_name", "") for m in _HEADING_FONT_MARKERS):
                pending.append(text)
            else:
                if pending:
                    headings.append(_join_parts(pending))
                    pending = []

    if pending:
        headings.append(_join_parts(pending))

    return headings


def _join_parts(parts: list[str]) -> str:
    """Join heading token parts; '.' prefix means continuation of a section number."""
    result = parts[0]
    for part in parts[1:]:
        result += part if part.startswith(".") else " " + part
    return result


class HeadingHierarchy:
    """
    Reconstructs heading ancestry across sequential chunks.

    Parsers return only the immediate heading per chunk, not the full ancestor
    chain. This class maintains a level-indexed state and infers ancestry from
    the heading level — either supplied explicitly (markdown # count) or inferred
    from section number prefixes (e.g. "1.2.1" → level 3, ancestors "1", "1.2").

    Eviction rules (section-number mode only):
      - Numbered entries that are not numeric ancestors of the current heading
        are evicted.
      - Non-numbered entries (e.g. "Preface") are evicted when a numbered
        heading is pushed — they are frontmatter, not structural ancestors.
    """

    def __init__(self) -> None:
        self._levels: dict[int, str] = {}

    def update(self, heading: str, level: int | None = None) -> None:
        """
        Advance the hierarchy state with the next observed heading.

        Args:
            heading: Heading text.
            level:   Explicit level (e.g. markdown # count: # → 1, ## → 2).
                     If omitted, level and ancestry are inferred from the
                     section number prefix in the heading text.
        """
        explicit = level is not None
        level = level if explicit else _heading_level(heading)

        for l in [l for l in self._levels if l >= level]:
            del self._levels[l]

        if not explicit:
            ancestors = _ancestor_numbers(heading)
            is_numbered = bool(_section_number(heading))
            for l in list(self._levels):
                entry_num = _section_number(self._levels[l])
                if (entry_num and entry_num not in ancestors) or (not entry_num and is_numbered):
                    del self._levels[l]

        self._levels[level] = heading

    @property
    def path(self) -> str:
        """Full heading path: 'Chapter => Section => Subsection'."""
        return " => ".join(self._levels[l] for l in sorted(self._levels))

    @property
    def parent_path(self) -> str:
        """Ancestor path excluding the innermost heading."""
        levels = sorted(self._levels)
        return " => ".join(self._levels[l] for l in levels[:-1])


def _section_number(heading: str) -> str:
    """Extract section number prefix if present (e.g. '1.2.1 Foo' → '1.2.1')."""
    token = heading.split()[0] if heading.split() else ""
    return token if re.match(r"^\d+(\.\d+)*$", token) else ""


def _heading_level(heading: str) -> int:
    """Level from section number dot-count (e.g. '1.2.1' → 3). Non-numbered → 1."""
    num = _section_number(heading)
    return num.count(".") + 1 if num else 1


def _ancestor_numbers(heading: str) -> set[str]:
    """All ancestor section numbers (e.g. '1.2.1' → {'1', '1.2'})."""
    num = _section_number(heading)
    if not num:
        return set()
    parts = num.split(".")
    return {".".join(parts[:i + 1]) for i in range(len(parts) - 1)}
