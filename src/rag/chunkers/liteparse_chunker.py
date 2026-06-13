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
            if para_tokens > CHUNK_MAX_TOKENS:
                if current:
                    chunks.append(self._make_chunk(parse_result, "\n\n".join(current)))
                    current = []
                    current_tokens = 0
                for seg in self._split_sentences(para):
                    chunks.append(self._make_chunk(parse_result, seg))
            elif current_tokens + para_tokens > CHUNK_MAX_TOKENS:
                chunks.append(self._make_chunk(parse_result, "\n\n".join(current)))
                current = [para]
                current_tokens = para_tokens
            else:
                current.append(para)
                current_tokens += para_tokens

        if current:
            chunks.append(self._make_chunk(parse_result, "\n\n".join(current)))

        return chunks, {}

    def _split_sentences(self, text: str) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(self._tokenizer.encode(sent))
            if current_tokens + sent_tokens > CHUNK_MAX_TOKENS and current:
                result.append(" ".join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            result.append(" ".join(current))

        return result or [text]

    def _make_chunk(self, parse_result: ParseResult, text: str) -> Chunk:
        return Chunk(
            text=text,
            enriched_text=text,
            headings="",
            parent_headings="",
            content_hash=parse_result.content_hash,
            filename=parse_result.source_path.rsplit("/", 1)[-1],
        )
