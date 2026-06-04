import hashlib
from pathlib import Path

from docling.document_converter import DocumentConverter

from rag.core.models import ParseResult
from rag.config import DATA_DIR

_MIN_CONTENT_LENGTH = 500  # chars; below this likely means a scanned/empty PDF


def _stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("-", "_").lower()


class DoclingParser:
    """Parses PDF, DOCX, and PPTX files to markdown via docling."""

    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def parse(self, source: Path | str) -> ParseResult:
        source = Path(source)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        result = self._converter.convert(str(source))
        md_text = result.document.export_to_markdown()

        stem = _stem(source)
        out_file = DATA_DIR / f"{stem}_converted.md"
        out_file.write_text(md_text, encoding="utf-8")
        print(f"[parser] Saved markdown: {out_file}")

        # Docling returns a uint64 which can exceed int64 max; store as hex string
        doc_dict = result.document.export_to_dict()
        raw_hash = doc_dict.get("origin", {}).get("binary_hash")
        if raw_hash:
            content_hash = hex(int(raw_hash))
        else:
            content_hash = hashlib.sha256(source.read_bytes()).hexdigest()

        if len(md_text.strip()) < _MIN_CONTENT_LENGTH:
            raise ValueError(
                f"Parsed content is suspiciously short ({len(md_text)} chars). "
                f"'{source.name}' may be a scanned PDF — consider OCR preprocessing."
            )

        suffix = source.suffix.lower().lstrip(".")
        return ParseResult(
            content=md_text,
            content_hash=content_hash,
            source_path=str(source),
            content_type=suffix,
        )
