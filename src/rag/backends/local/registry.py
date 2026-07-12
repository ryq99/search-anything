import json

from rag.core.schemas import BookEntry
from rag.config import PROCESSED_BOOKS_PATH


class JsonRegistry:
    """Book ingestion ledger backed by a local JSON file."""

    def is_ingested(self, content_hash: str, pipeline_config_hash: str) -> bool:
        return f"{content_hash}_{pipeline_config_hash}" in self._load().get("books", {})

    def register(self, entry: BookEntry) -> None:
        data = self._load()
        data.setdefault("books", {})[entry.registry_key] = entry.to_dict()
        PROCESSED_BOOKS_PATH.write_text(json.dumps(data, indent=2))

    def get(self, content_hash: str, pipeline_config_hash: str) -> dict:
        return self._load().get("books", {}).get(f"{content_hash}_{pipeline_config_hash}", {})

    def find_superseded(self, filename: str, pipeline_config_hash: str, content_hash: str) -> list[str]:
        # TODO(local supersede follow-up): return prior versions and wire Milvus
        # vector deletion. Kept as a no-op so local behavior is unchanged for now.
        return []

    def delete(self, content_hash: str, pipeline_config_hash: str) -> None:
        data = self._load()
        data.get("books", {}).pop(f"{content_hash}_{pipeline_config_hash}", None)
        PROCESSED_BOOKS_PATH.write_text(json.dumps(data, indent=2))

    def _load(self) -> dict:
        if PROCESSED_BOOKS_PATH.exists():
            return json.loads(PROCESSED_BOOKS_PATH.read_text())
        return {"books": {}}
