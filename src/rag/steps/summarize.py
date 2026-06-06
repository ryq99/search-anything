import asyncio
from pathlib import Path

import pandas as pd

from rag.core.interfaces import LLM
from rag.config import DATA_DIR, SUMMARY_SEMAPHORE, SUMMARY_MAX_TOKENS

_SYSTEM = (
    "Summarize the book contents with condensed form in 3-4 sentences. "
    "Cover the main topics, definition, proof, methods, and applications if any."
)


async def _summarize_one(
    llm: LLM,
    semaphore: asyncio.Semaphore,
    parent_headings: str,
    text: str,
) -> tuple[str, str]:
    async with semaphore:
        summary = await llm.acomplete(
            system=_SYSTEM,
            user=f"Summarize the following text:\n\n{text}",
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        return parent_headings, summary


async def summarize_all(
    parent_headings_text: dict[str, str],
    llm: LLM,
) -> dict[str, str]:
    """Concurrently summarize all parent heading groups using the provided LLM."""
    semaphore = asyncio.Semaphore(SUMMARY_SEMAPHORE)
    tasks = [
        _summarize_one(llm, semaphore, ph, text)
        for ph, text in parent_headings_text.items()
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)


def save_summaries_csv(summaries: dict[str, str], stem: str) -> Path:
    out_file = DATA_DIR / f"{stem}_parent_headings_summary.csv"
    pd.DataFrame(
        summaries.items(), columns=["Parent Headings", "Summary"]
    ).to_csv(out_file, index=False)
    print(f"[summarize] Saved summaries: {out_file}")
    return out_file
