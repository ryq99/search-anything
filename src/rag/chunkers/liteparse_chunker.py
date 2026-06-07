import re
from pathlib import Path

from rag.core.schemas import Chunk, ParseResult
from rag.config import CHUNK_MAX_TOKENS, CHUNK_TOKENIZER


class LiteParseChunker:
    """
    Markdown-native chunker for LiteParseParser output. No docling dependency.

    Pipeline:
      1. Split on markdown headings (#, ##, ###...) — structural boundary pass
      2. Token-aware split — break any section exceeding max_tokens by paragraph
      3. Enrich — prepend full heading path to each chunk text for embedding

    Enrichment mirrors docling's contextualize() output so the embedding input
    format is consistent across both parser paths.
    """

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER)

    def chunk(self, parse_result: ParseResult) -> tuple[list[Chunk], dict[str, str]]:
        sections = _split_by_headings(parse_result.markdown)

        chunks: list[Chunk] = []
        parent_headings_text: dict[str, str] = {}

        for heading_stack, text in sections:
            headings_str = " => ".join(heading_stack)
            parent_headings_str = " => ".join(heading_stack[:-1]) if len(heading_stack) > 1 else ""

            for sub_text in self._split_by_tokens(text):
                chunk = Chunk(
                    text=sub_text,
                    enriched_text=_contextualize(heading_stack, sub_text),
                    headings=headings_str,
                    parent_headings=parent_headings_str,
                    content_hash=parse_result.content_hash,
                    filename=parse_result.source_path.rsplit("/", 1)[-1],
                )
                chunks.append(chunk)

            if parent_headings_str:
                if parent_headings_str not in parent_headings_text:
                    parent_headings_text[parent_headings_str] = text
                else:
                    parent_headings_text[parent_headings_str] += "\n" + text

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


def _split_by_headings(markdown: str) -> list[tuple[list[str], str]]:
    """
    Split markdown into (heading_stack, content) pairs.

    heading_stack is the ordered list of ancestor headings at that point,
    e.g. ["3 Supervised Learning", "3.2 Linear Regression"].
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[str] = []
    last_pos = 0

    for match in heading_re.finditer(markdown):
        level = len(match.group(1))
        title = match.group(2).strip()

        text_before = markdown[last_pos:match.start()].strip()
        if text_before:
            sections.append((list(heading_stack), text_before))

        # Trim stack to parent level then push current heading
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(title)
        last_pos = match.end()

    remaining = markdown[last_pos:].strip()
    if remaining:
        sections.append((list(heading_stack), remaining))

    return [(stack, text) for stack, text in sections if text.strip()]


def _contextualize(heading_stack: list[str], text: str) -> str:
    """Prepend heading path to text, mirroring docling HybridChunker.contextualize()."""
    if not heading_stack:
        return text
    return " => ".join(heading_stack) + "\n\n" + text
