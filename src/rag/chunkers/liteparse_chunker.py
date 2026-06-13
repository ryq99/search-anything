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
        # split the full markdown into paragraphs on blank lines; skip empty segments
        paragraphs = [p.strip() for p in re.split(r"\n\n+", parse_result.markdown) if p.strip()]
        chunks: list[Chunk] = []
        current: list[str] = []   # paragraphs staged for the next chunk
        current_tokens = 0        # running token count of staged paragraphs

        for para in paragraphs:
            para_tokens = len(self._tokenizer.encode(para))

            if para_tokens > CHUNK_MAX_TOKENS:
                # paragraph alone exceeds the limit — flush what's staged, then
                # fall back to sentence-level splitting for this paragraph
                if current:
                    chunks.append(self._make_chunk(parse_result, "\n\n".join(current)))
                    current = []
                    current_tokens = 0
                for seg in self._split_sentences(para):
                    chunks.append(self._make_chunk(parse_result, seg))

            elif current_tokens + para_tokens > CHUNK_MAX_TOKENS:
                # adding this paragraph would overflow — flush staged paragraphs
                # as a chunk, then start a fresh accumulator with this paragraph
                chunks.append(self._make_chunk(parse_result, "\n\n".join(current)))
                current = [para]
                current_tokens = para_tokens

            else:
                # paragraph fits — keep accumulating
                current.append(para)
                current_tokens += para_tokens

        # emit any remaining staged paragraphs as the final chunk
        if current:
            chunks.append(self._make_chunk(parse_result, "\n\n".join(current)))

        return chunks, {}  # empty dict: no heading groups to summarize

    def _split_sentences(self, text: str) -> list[str]:
        # split on sentence-ending punctuation followed by whitespace
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        result: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(self._tokenizer.encode(sent))
            if current_tokens + sent_tokens > CHUNK_MAX_TOKENS and current:
                # flush staged sentences and start fresh
                result.append(" ".join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            result.append(" ".join(current))

        # if even a single sentence exceeds the limit, emit it as-is rather than
        # breaking mid-sentence
        return result or [text]

    def _make_chunk(self, parse_result: ParseResult, text: str) -> Chunk:
        return Chunk(
            text=text,
            enriched_text=text,  # no heading prefix for liteparse chunks
            headings="",
            parent_headings="",
            content_hash=parse_result.content_hash,
            filename=parse_result.source_path.rsplit("/", 1)[-1],
        )
