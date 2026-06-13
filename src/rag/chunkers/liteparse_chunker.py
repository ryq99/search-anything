import re

from rag.core.schemas import Chunk, ParseResult
from rag.config import CHUNK_MAX_TOKENS, CHUNK_TOKENIZER


class LiteParseChunker:
    """
    Paragraph-based chunker for LiteParseParser output.

    Splits on double-newline paragraph boundaries, merging consecutive
    paragraphs until CHUNK_MAX_TOKENS is reached. No heading structure
    is inferred — liteparse produces layout-preserving markdown without
    reliable semantic markers, so heading fields are left empty.
    """

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER)

    def chunk(self, parse_result: ParseResult) -> tuple[list[Chunk], dict[str, str]]:
        paragraphs = [p.strip() for p in re.split(r"\n\n+", parse_result.markdown) if p.strip()]
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self._tokenizer.encode(para))
            if current_tokens + para_tokens > CHUNK_MAX_TOKENS and current:
                chunks.append(self._make_chunk(parse_result, "\n\n".join(current)))
                current = [para]
                current_tokens = para_tokens
            else:
                current.append(para)
                current_tokens += para_tokens

        if current:
            chunks.append(self._make_chunk(parse_result, "\n\n".join(current)))

        return chunks, {}

    def _make_chunk(self, parse_result: ParseResult, text: str) -> Chunk:
        return Chunk(
            text=text,
            enriched_text=text,
            headings="",
            parent_headings="",
            content_hash=parse_result.content_hash,
            filename=parse_result.source_path.rsplit("/", 1)[-1],
        )
