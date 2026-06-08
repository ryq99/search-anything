import re

from rag.core.schemas import Chunk, ParseResult
from rag.parsers.heading_hierarchy import HeadingHierarchy
from rag.config import CHUNK_MAX_TOKENS, CHUNK_TOKENIZER


class LiteParseChunker:
    """
    Markdown-native chunker for LiteParseParser output. No docling dependency.

    Pipeline:
      1. Split on markdown headings (#, ##, ###...) — structural boundary pass
      2. Token-aware split — break any section exceeding max_tokens by paragraph
      3. Enrich — prepend full heading path to each chunk text for embedding

    Heading hierarchy is tracked via HeadingHierarchy (explicit # level),
    consistent with how DoclingChunker tracks it via section number inference.
    Enrichment mirrors docling's contextualize() output so the embedding input
    format is consistent across both parser paths.
    """

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER)

    def chunk(self, parse_result: ParseResult) -> tuple[list[Chunk], dict[str, str]]:
        hierarchy = HeadingHierarchy()
        chunks: list[Chunk] = []
        parent_headings_text: dict[str, str] = {}

        for level, title, text in _split_by_headings(parse_result.markdown):
            if title:
                hierarchy.update(title, level=level)

            for sub_text in self._split_by_tokens(text):
                chunk = Chunk(
                    text=sub_text,
                    enriched_text=_contextualize(hierarchy.path, sub_text),
                    headings=hierarchy.path,
                    parent_headings=hierarchy.parent_path,
                    content_hash=parse_result.content_hash,
                    filename=parse_result.source_path.rsplit("/", 1)[-1],
                )
                chunks.append(chunk)

            if hierarchy.parent_path:
                parent_headings_text[hierarchy.parent_path] = (
                    parent_headings_text.get(hierarchy.parent_path, "") + "\n" + text
                ).lstrip("\n")

        return chunks, parent_headings_text

    def _split_by_tokens(self, text: str) -> list[str]:
        """Split text into token-bounded segments by paragraph boundaries."""
        if len(self._tokenizer.encode(text)) <= CHUNK_MAX_TOKENS:
            return [text]

        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self._tokenizer.encode(para))
            if current_tokens + para_tokens > CHUNK_MAX_TOKENS and current:
                result.append("\n\n".join(current))
                current = [para]
                current_tokens = para_tokens
            else:
                current.append(para)
                current_tokens += para_tokens

        if current:
            result.append("\n\n".join(current))

        return result or [text]


def _split_by_headings(markdown: str) -> list[tuple[int, str, str]]:
    """
    Split markdown into (level, heading_title, content) triples.

    level=0 and title='' for content that appears before the first heading.
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[int, str, str]] = []
    current_level, current_title, last_pos = 0, "", 0

    for match in heading_re.finditer(markdown):
        text_before = markdown[last_pos:match.start()].strip()
        if text_before:
            sections.append((current_level, current_title, text_before))
        current_level = len(match.group(1))
        current_title = match.group(2).strip()
        last_pos = match.end()

    remaining = markdown[last_pos:].strip()
    if remaining:
        sections.append((current_level, current_title, remaining))

    return [(l, t, s) for l, t, s in sections if s]


def _contextualize(path: str, text: str) -> str:
    """Prepend heading path to text, mirroring docling HybridChunker.contextualize()."""
    return f"{path}\n\n{text}" if path else text
