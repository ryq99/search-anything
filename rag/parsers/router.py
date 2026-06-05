from pathlib import Path

from rag.core.schemas import ParseResult

# Extensions handled by each parser family
_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx"}
_PASSTHROUGH_EXTENSIONS = {".md", ".txt"}

SUPPORTED_EXTENSIONS = _DOCLING_EXTENSIONS | _PASSTHROUGH_EXTENSIONS


def detect_content_type(source: Path | str) -> str:
    source_str = str(source)
    if source_str.startswith(("http://", "https://")):
        if "youtube.com" in source_str or "youtu.be" in source_str:
            return "youtube"
        return "web"
    ext = Path(source).suffix.lower()
    return {
        ".pdf":  "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
        ".ipynb": "notebook",
        ".md":   "markdown",
        ".txt":  "text",
    }.get(ext, "unknown")


def parse_to_markdown(source: Path | str) -> ParseResult:
    """Detect content type and route to the appropriate parser."""
    content_type = detect_content_type(source)

    if content_type in ("pdf", "docx", "pptx"):
        from rag.parsers.docling_parser import DoclingParser
        return DoclingParser().parse(source)

    if content_type in ("markdown", "text"):
        return _parse_plaintext(source)

    raise NotImplementedError(
        f"No parser registered for content type '{content_type}'. "
        f"Supported: pdf, docx, pptx, markdown, text. "
        f"Coming soon: web, notebook, youtube."
    )


def _parse_plaintext(source: Path | str) -> ParseResult:
    import hashlib
    source = Path(source)
    content = source.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    return ParseResult(
        content=content,
        content_hash=content_hash,
        source_path=str(source),
        content_type="text",
    )
