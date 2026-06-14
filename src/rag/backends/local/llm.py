import asyncio

import anthropic
import httpx

from rag.config import (
    ANTHROPIC_API_KEY, API_LLM_MODEL,
    LOCAL_LLM_MODEL, LOCAL_LLM_BASE_URL,
)


class AnthropicLLM:
    """LLM backend using the Anthropic SDK. Used for synthesis (query-time answer generation)."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._async_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=API_LLM_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text or ""

    async def acomplete(self, system: str, user: str, max_tokens: int) -> str:
        for attempt in range(5):
            try:
                response = await self._async_client.messages.create(
                    model=API_LLM_MODEL,
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


class LocalLLM:
    """LLM backend for chunk summarization (indexing-time). Calls Ollama via httpx.

    Structured identically to AnthropicLLM so it can be swapped for BedrockLLM
    in cloud/prod without changing any call-site code.
    """

    def __init__(self) -> None:
        self._async_client = httpx.AsyncClient(base_url=LOCAL_LLM_BASE_URL, timeout=120)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        return asyncio.run(self.acomplete(system, user, max_tokens))

    async def acomplete(self, system: str, user: str, max_tokens: int) -> str:
        response = await self._async_client.post(
            "/api/chat",
            json={
                "model": LOCAL_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"] or ""
