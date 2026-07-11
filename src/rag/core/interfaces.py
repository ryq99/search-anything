from typing import Protocol, runtime_checkable
from pathlib import Path

from rag.core.schemas import ParseResult, BookEntry


@runtime_checkable
class Parser(Protocol):
    """Converts a raw source (file path or URL) into markdown + content hash."""
    def parse(self, source: "Path | str") -> ParseResult: ...


class Registry(Protocol):
    """Tracks which sources have already been ingested. Backend-specific implementation."""
    def is_ingested(self, content_hash: str, pipeline_config_hash: str) -> bool: ...
    def register(self, entry: BookEntry) -> None: ...
    def get(self, content_hash: str, pipeline_config_hash: str) -> dict: ...


class LLM(Protocol):
    """Text completion. Sync for single calls, async for concurrent batch calls."""
    def complete(self, system: str, user: str, max_tokens: int) -> str: ...
    async def acomplete(self, system: str, user: str, max_tokens: int) -> str: ...
