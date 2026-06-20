import asyncio

from rag.core.schemas import Chunk
from rag.config import SUMMARY_SEMAPHORE, SUMMARY_MAX_TOKENS

_SYSTEM = (
    "Summarize the following text in 1-2 sentences covering the main point."
)


async def _summarize_chunk(llm, semaphore: asyncio.Semaphore, chunk: Chunk) -> str:
    async with semaphore:
        return await llm.acomplete(
            system=_SYSTEM,
            user=chunk.text,
            max_tokens=SUMMARY_MAX_TOKENS,
        )


async def summarize_chunks(chunks: list[Chunk], llm) -> None:
    """Concurrently fill chunk.summary in-place for all chunks."""
    semaphore = asyncio.Semaphore(SUMMARY_SEMAPHORE)
    summaries = await asyncio.gather(
        *[_summarize_chunk(llm, semaphore, c) for c in chunks]
    )
    for chunk, summary in zip(chunks, summaries):
        chunk.summary = summary
