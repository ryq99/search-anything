"""
Parsing step: the ingestion boundary of the pipeline.

Owns orchestration only — detects content type and routes to the appropriate
parser plugin in rag/parsers/. Each parser is responsible for its own artifact
saving and returns a ParseResult.
"""
from pathlib import Path

from rag.core.schemas import ParseResult
from rag.config import LOCAL_PARSER

_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}
_PASSTHROUGH_EXTENSIONS = {".md", ".txt"}
SUPPORTED_EXTENSIONS = _DOCUMENT_EXTENSIONS | _PASSTHROUGH_EXTENSIONS


def _get_document_parser():
    if LOCAL_PARSER == "liteparse":
        from rag.parsers.liteparse_parser import LiteParseParser
        return LiteParseParser()
    from rag.parsers.docling_parser import DoclingParser
    return DoclingParser()


def detect_content_type(source: Path | str) -> str:
    source_str = str(source)
    if source_str.startswith(("http://", "https://")):
        return "youtube" if ("youtube.com" in source_str or "youtu.be" in source_str) else "web"
    return {
        ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx",
        ".ipynb": "notebook", ".md": "markdown", ".txt": "text",
    }.get(Path(source).suffix.lower(), "unknown")


def parse_document(source: Path | str) -> ParseResult:
    content_type = detect_content_type(source)
    if content_type in ("pdf", "docx", "pptx"):
        return _get_document_parser().parse(source)
    if content_type in ("markdown", "text"):
        from rag.parsers.plaintext_parser import PlaintextParser
        return PlaintextParser().parse(source)
    raise NotImplementedError(
        f"No parser for content type '{content_type}'. "
        f"Supported: pdf, docx, pptx, markdown, text. Coming soon: web, notebook, youtube."
    )
