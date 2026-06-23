from datetime import datetime, timezone
from pathlib import Path

from rag.backends.factory import get_backend
from rag.core.schemas import BookEntry


def build(parse_result, chunks, backend) -> BookEntry:
    """Embed, store, and register chunks. Returns the BookEntry."""
    backend.vectorstore.store(chunks)
    entry = BookEntry(
        filename=Path(parse_result.source_path).name,
        source_path=parse_result.source_path,
        content_hash=parse_result.content_hash,
        parser=parse_result.parser,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        chunk_count=len(chunks),
    )
    backend.registry.register(entry)
    return entry
