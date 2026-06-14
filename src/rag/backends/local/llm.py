import asyncio

import anthropic
import openai

from rag.config import (
    ANTHROPIC_API_KEY, API_LLM_MODEL,
    LOCAL_LLM_MODEL, LOCAL_LLM_BASE_URL, LOCAL_LLM_API_KEY,
)


class AnthropicLLM:
    """LLM backend using the Anthropic SDK directly (local mode)."""

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
    """LLM backend via OpenAI-compatible API (Ollama or HuggingFace TGI). Used for chunk summarization."""

    def __init__(self) -> None:
        self._async_client = openai.AsyncOpenAI(
            base_url=LOCAL_LLM_BASE_URL,
            api_key=LOCAL_LLM_API_KEY,
        )

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        import asyncio
        return asyncio.run(self.acomplete(system, user, max_tokens))

    async def acomplete(self, system: str, user: str, max_tokens: int) -> str:
        response = await self._async_client.chat.completions.create(
            model=LOCAL_LLM_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
