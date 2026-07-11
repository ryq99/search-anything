from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParseResult:
    markdown: str
    content_hash: str   # SHA-256 hex string of the raw source file bytes
    source_path: str    # original file path or URL
    content_type: str   # "pdf", "markdown", "text", "web", "notebook", etc.
    parser: str = ""          # parser used: "docling", "liteparse", "plaintext"
    docling_document: Any = None  # docling: structured semantic tree (DoclingDocument)
    liteparse_pages: Any = None   # liteparse: page-level data with font metadata (list[ParsedPage])
    doc_dir: Path | None = None   # data/{stem}_{hash[:12]}_{parser}/


@dataclass
class Chunk:
    text: str            # raw chunk text (for display / summarization)
    enriched_text: str   # heading-prefixed text for embedding (contextualize output)
    headings: str        # full heading path: "1 Intro => 1.2 Motivation"
    parent_headings: str # ancestor headings (all but innermost)
    content_hash: str    # hex string — ties chunk back to its source document
    filename: str
    summary: str = ""    # 1-2 sentence summary filled by summarize step
    page_numbers: list[int] = field(default_factory=list)  # source pages this chunk spans (docling only)


@dataclass
class BookEntry:
    filename: str
    source_path: str
    content_hash: str
    pipeline_config_hash: str  # fingerprint of all method decisions (parse→chunk→summarize→embed)
    parser: str
    ingested_at: str
    chunk_count: int

    @property
    def registry_key(self) -> str:
        return f"{self.content_hash}_{self.pipeline_config_hash}"

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "source_path": self.source_path,
            "content_hash": self.content_hash,
            "parser": self.parser,
            "pipeline_config_hash": self.pipeline_config_hash,
            "ingested_at": self.ingested_at,
            "chunk_count": self.chunk_count,
        }
