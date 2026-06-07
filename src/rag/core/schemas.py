from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParseResult:
    markdown: str
    content_hash: str   # SHA-256 hex string of the raw source file bytes
    source_path: str    # original file path or URL
    content_type: str   # "pdf", "docx", "web", "notebook", etc.
    docling_document: Any = None # docling structured parse result (semantic tree)
    doc_dir: Path | None = None # per-document data folder for outputs


@dataclass
class Chunk:
    text: str            # raw chunk text (for display / summarization)
    enriched_text: str   # heading-prefixed text for embedding (contextualize output)
    headings: str        # full heading path: "1 Intro => 1.2 Motivation"
    parent_headings: str # ancestor headings (all but innermost)
    content_hash: str    # hex string — ties chunk back to its source document
    filename: str


@dataclass
class BookEntry:
    filename: str
    source_path: str
    content_hash: str
    ingested_at: str
    chunk_count: int
    summary_artifact_path: str = ""

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "source_path": self.source_path,
            "content_hash": self.content_hash,
            "ingested_at": self.ingested_at,
            "chunk_count": self.chunk_count,
            "summary_artifact_path": self.summary_artifact_path,
        }
