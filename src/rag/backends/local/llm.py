import asyncio

import anthropic
import httpx

from rag.config import ANTHROPIC_API_KEY, LOCAL_LLM_BASE_URL


class LocalLLM:
    """LLM backend via Ollama (httpx). Used in local mode for summarization and synthesis."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._async_client = httpx.AsyncClient(base_url=LOCAL_LLM_BASE_URL, timeout=120)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        return asyncio.run(self.acomplete(system, user, max_tokens))

    async def acomplete(self, system: str, user: str, max_tokens: int) -> str:
        response = await self._async_client.post(
            "/api/chat",
            json={
                "model": self._model,
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


class AnthropicLLM:
    """LLM backend via Anthropic API. Used in cloud mode for summarization and synthesis."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._async_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text or ""

    async def acomplete(self, system: str, user: str, max_tokens: int) -> str:
        for attempt in range(5):
            try:
                response = await self._async_client.messages.create(
                    model=self._model,
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
