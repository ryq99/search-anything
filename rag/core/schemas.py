from dataclasses import dataclass


@dataclass
class ParseResult:
    content: str        # markdown text
    content_hash: str   # hex string (from docling or SHA-256)
    source_path: str    # original file path or URL
    content_type: str   # "pdf", "docx", "web", "notebook", etc.


@dataclass
class BookEntry:
    filename: str
    source_path: str
    content_hash: str
    ingested_at: str
    chunk_count: int
    toc_artifact_path: str = ""
    summary_artifact_path: str = ""

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "source_path": self.source_path,
            "content_hash": self.content_hash,
            "ingested_at": self.ingested_at,
            "chunk_count": self.chunk_count,
            "toc_artifact_path": self.toc_artifact_path,
            "summary_artifact_path": self.summary_artifact_path,
        }
