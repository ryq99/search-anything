import json

from rag.core.schemas import BookEntry
from rag.config import REGISTRY_PATH


class JsonRegistry:
    """Book ingestion ledger backed by a local JSON file."""

    def is_ingested(self, content_hash: str, pipeline_config_hash: str) -> bool:
        return f"{content_hash}_{pipeline_config_hash}" in self._load().get("books", {})

    def register(self, entry: BookEntry) -> None:
        data = self._load()
        data.setdefault("books", {})[entry.registry_key] = entry.to_dict()
        self._save(data)

    def get(self, content_hash: str, pipeline_config_hash: str) -> dict:
        return self._load().get("books", {}).get(f"{content_hash}_{pipeline_config_hash}", {})

    def find_superseded(self, filename: str, pipeline_config_hash: str, content_hash: str) -> list[str]:
        """Older versions of the same document: same filename + config, different content."""
        return [
            e["content_hash"]
            for e in self._load().get("books", {}).values()
            if e.get("filename") == filename
            and e.get("pipeline_config_hash") == pipeline_config_hash
            and e.get("content_hash") != content_hash
        ]

    def delete(self, content_hash: str, pipeline_config_hash: str) -> None:
        data = self._load()
        data.get("books", {}).pop(f"{content_hash}_{pipeline_config_hash}", None)
        self._save(data)

    def _load(self) -> dict:
        if REGISTRY_PATH.exists():
            return json.loads(REGISTRY_PATH.read_text())
        return {"books": {}}

    def _save(self, data: dict) -> None:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(json.dumps(data, indent=2))
