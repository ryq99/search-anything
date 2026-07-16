import asyncio
import re

from rag.core.schemas import Chunk
from rag.config import SUMMARY_SEMAPHORE, SUMMARY_MAX_TOKENS

_SYSTEM = (
    "Summarize the following text in 1-2 sentences covering the main point. "
    "Return only the summary itself — no headings, labels, preamble, or markdown formatting."
)


def _clean(summary: str) -> str:
    """Strip a leading 'Summary' heading/label the model sometimes adds despite
    the prompt (e.g. '# Summary\\n\\n', '**Summary:**', 'Summary:')."""
    summary = re.sub(r"^\s*#{1,6}\s*summary\s*\n+", "", summary, flags=re.IGNORECASE)
    summary = re.sub(r"^\s*\**\s*summary\s*\**\s*:\s*\**\s*", "", summary, flags=re.IGNORECASE)
    return summary.strip()


async def _summarize_chunk(llm, semaphore: asyncio.Semaphore, chunk: Chunk) -> str:
    async with semaphore:
        try:
            return _clean(await llm.acomplete(
                system=_SYSTEM,
                user=chunk.text,
                max_tokens=SUMMARY_MAX_TOKENS,
            ))
        except Exception as e:
            # A single failed summary must not abort the whole document's gather().
            # Fall back to truncated raw text so the chunk still carries usable context.
            print(f"[summarize] Summary failed ({type(e).__name__}); using truncated text fallback.")
            return chunk.text[:300]


async def summarize_chunks(chunks: list[Chunk], llm) -> None:
    """Concurrently fill chunk.summary in-place for all chunks."""
    semaphore = asyncio.Semaphore(SUMMARY_SEMAPHORE)
    summaries = await asyncio.gather(
        *[_summarize_chunk(llm, semaphore, c) for c in chunks]
    )
    for chunk, summary in zip(chunks, summaries):
        chunk.summary = summary
