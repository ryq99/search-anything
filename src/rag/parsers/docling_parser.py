import hashlib
import json
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

        # Hash the raw file bytes first — parser-agnostic, stable folder name
        content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        doc_dir = DATA_DIR / f"{_stem(source)}_{content_hash[:12]}_docling"
        parse_dir = doc_dir / "parse"
        parse_dir.mkdir(parents=True, exist_ok=True)

        result = self._converter.convert(str(source))
        md_text = _clean_markdown(_export_markdown(result.document))

        # Validate BEFORE writing, so failed parses leave no orphan files behind
        if len(md_text) < PARSER_MIN_CONTENT_LENGTH:
            raise ValueError(
                f"Parsed content is suspiciously short ({len(md_text)} chars). "
                f"'{source.name}' may be a scanned PDF — enable OCR "
                f"(PARSER_ENABLE_OCR=true) or pre-process with OCR."
            )

        # converted.md — human-readable audit artifact; fallback chunking input
        (parse_dir / "converted.md").write_text(md_text, encoding="utf-8")
        print(f"[parser:docling] Saved markdown: {parse_dir / 'converted.md'}")

        # docling.json — full structured document; enables re-chunking without re-parsing
        (parse_dir / "docling.json").write_text(
            json.dumps(result.document.export_to_dict()), encoding="utf-8"
        )
        print(f"[parser:docling] Saved docling document: {parse_dir / 'docling.json'}")

        # parse_result.json — ParseResult metadata; ties all parse artifacts together
        parse_result_meta = {
            "content_hash": content_hash,
            "source_path": str(source),
            "content_type": source.suffix.lower().lstrip("."),
        }
        (parse_dir / "parse_result.json").write_text(
            json.dumps(parse_result_meta, indent=2), encoding="utf-8"
        )
        print(f"[parser:docling] Saved parse result: {parse_dir / 'parse_result.json'}")

        return ParseResult(
            markdown=md_text,
            content_hash=content_hash,
            source_path=str(source),
            content_type=source.suffix.lower().lstrip("."),
            parser="docling",
            docling_document=result.document,
            doc_dir=doc_dir,
        )
