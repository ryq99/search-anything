from datetime import datetime, timezone
from pathlib import Path

from rag.core.schemas import BookEntry
from rag.config import PIPELINE_CONFIG_HASH


def build(parse_result, chunks, backend) -> BookEntry:
    """Embed, store, and register chunks. Returns the BookEntry.

    Supersedes any older version of the same document (same filename + config,
    different content) so a query never sees the old and new version at once.
    The vectorstore removes the old chunks in the same sync that adds the new
    ones; the registry rows are dropped only after that sync succeeds.
    """
    filename = Path(parse_result.source_path).name
    superseded = backend.registry.find_superseded(
        filename, PIPELINE_CONFIG_HASH, parse_result.content_hash
    )

    backend.vectorstore.store(chunks, superseded=superseded)

    for old_content_hash in superseded:
        backend.registry.delete(old_content_hash, PIPELINE_CONFIG_HASH)

    entry = BookEntry(
        filename=filename,
        source_path=parse_result.source_path,
        content_hash=parse_result.content_hash,
        pipeline_config_hash=PIPELINE_CONFIG_HASH,
        parser=parse_result.parser,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        chunk_count=len(chunks),
    )
    backend.registry.register(entry)
    return entry
