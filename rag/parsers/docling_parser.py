import hashlib
import re
from pathlib import Path

from rag.core.schemas import ParseResult
from rag.config import DATA_DIR, PARSER_ENABLE_OCR, PARSER_MIN_CONTENT_LENGTH


def _stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("-", "_").lower()


def _build_converter():
    """
    Build a docling DocumentConverter with a tuned PDF pipeline.

    - OCR on (force_full_page_ocr=False): clean text-layer pages use fast native
      extraction; only pages where text extraction fails invoke OCR. Recovers
      content from partially-scanned books without slowing down clean PDFs.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = PARSER_ENABLE_OCR
    if PARSER_ENABLE_OCR:
        # Only OCR pages that fail native extraction — keeps text-layer PDFs fast
        try:
            from docling.datamodel.pipeline_options import EasyOcrOptions
            pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)
        except ImportError:
            # easyocr not installed; fall back to docling's default OCR engine
            pass
    # Keep table structure so export can render real markdown tables
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _export_markdown(document) -> str:
    """Export to markdown, preserving tables as markdown tables (not flattened prose)."""
    try:
        from docling_core.types.doc import TableExportMode
        return document.export_to_markdown(table_export_mode=TableExportMode.MARKDOWN)
    except (ImportError, TypeError):
        # Older docling without TableExportMode arg — fall back to default export
        return document.export_to_markdown()


def _clean_markdown(md_text: str) -> str:
    """Collapse 3+ consecutive blank lines into 2 to reduce whitespace noise in chunks."""
    return re.sub(r"\n{3,}", "\n\n", md_text).strip()


class DoclingParser:
    """Parses PDF, DOCX, and PPTX files to markdown via docling (ML layout model)."""

    def __init__(self) -> None:
        self._converter = _build_converter()

    def parse(self, source: Path | str) -> ParseResult:
        source = Path(source)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        result = self._converter.convert(str(source))
        md_text = _clean_markdown(_export_markdown(result.document))

        # Validate BEFORE writing, so failed parses leave no orphan .md file behind
        if len(md_text) < PARSER_MIN_CONTENT_LENGTH:
            raise ValueError(
                f"Parsed content is suspiciously short ({len(md_text)} chars). "
                f"'{source.name}' may be a scanned PDF — enable OCR "
                f"(PARSER_ENABLE_OCR=true) or pre-process with OCR."
            )

        stem = _stem(source)
        out_file = DATA_DIR / f"{stem}_converted.md"
        out_file.write_text(md_text, encoding="utf-8")
        print(f"[parser:docling] Saved markdown: {out_file}")

        return ParseResult(
            content=md_text,
            content_hash=_content_hash(result.document, source),
            source_path=str(source),
            content_type=source.suffix.lower().lstrip("."),
        )


def _content_hash(document, source: Path) -> str:
    """Prefer docling's document hash (hex string); fall back to SHA-256 of the file bytes."""
    try:
        raw_hash = document.export_to_dict().get("origin", {}).get("binary_hash")
        if raw_hash:
            # Docling returns a uint64 which can exceed int64 max; store as hex string
            return hex(int(raw_hash))
    except Exception:
        pass
    return hashlib.sha256(source.read_bytes()).hexdigest()
