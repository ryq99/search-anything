import asyncio

import anthropic

from rag.config import ANTHROPIC_API_KEY, TOC_MODEL


class AnthropicLLM:
    """LLM backend using the Anthropic SDK directly (local mode)."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._async_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=TOC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text or ""

    async def acomplete(self, system: str, user: str, max_tokens: int) -> str:
        for attempt in range(5):
            try:
                response = await self._async_client.messages.create(
                    model=TOC_MODEL,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return response.content[0].text or ""
            except anthropic.RateLimitError:
                if attempt == 4:
                    raise
                wait = 60 * (2 ** attempt)
                print(f"[llm] Rate limited, retrying in {wait}s... (attempt {attempt + 1}/5)")
                await asyncio.sleep(wait)
        return ""  # unreachable but satisfies type checker
