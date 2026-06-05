from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParseResult:
    content: str        # markdown text (audit artifact + liteparse chunking input)
    content_hash: str   # hex string (from docling or SHA-256)
    source_path: str    # original file path or URL
    content_type: str   # "pdf", "docx", "web", "notebook", etc.
    # Populated by DoclingParser; None for liteparse/plaintext paths.
    # Typed as Any so core/ stays free of heavy docling imports.
    docling_document: Any = None


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
