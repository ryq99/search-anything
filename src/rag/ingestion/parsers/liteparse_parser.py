import dataclasses
import hashlib
import json
import re
from pathlib import Path

from rag.core.schemas import ParseResult
from rag.config import DATA_DIR, PARSER_ENABLE_OCR, PARSER_MIN_CONTENT_LENGTH


def _page_to_dict(page) -> dict:
    if dataclasses.is_dataclass(page) and not isinstance(page, type):
        return dataclasses.asdict(page)
    return vars(page) if hasattr(page, "__dict__") else str(page)

# File types liteparse handles natively (Rust core: PDF/images)
_LITEPARSE_EXTENSIONS = {".pdf"}


def _stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("-", "_").lower()


def _clean_markdown(md_text: str) -> str:
    """Collapse 3+ consecutive blank lines into 2 to reduce whitespace noise in chunks."""
    return re.sub(r"\n{3,}", "\n\n", md_text).strip()


class LiteParseParser:
    """
    Parses documents to markdown via LiteParse (LlamaIndex).

    Uses native Rust bindings — fast, in-process, zero heavy Python deps. Built-in
    Tesseract OCR for scanned pages. Philosophy is spatial layout *preservation*
    (project text onto a grid) rather than ML structure *detection* like docling,
    which is why it's worth A/B comparing on real books.
    """

    def __init__(self) -> None:
        from liteparse import LiteParse

        # output_format="markdown" yields layout-preserving markdown; OCR auto-fires
        # on scanned pages when enabled.
        self._parser = LiteParse(
            output_format="markdown",
            ocr_enabled=PARSER_ENABLE_OCR,
            quiet=True,
        )

    def parse(self, source: Path | str) -> ParseResult:
        source = Path(source)

        # Hash the raw file bytes first — parser-agnostic, stable folder name
        content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        doc_dir = DATA_DIR / f"{_stem(source)}_{content_hash[:12]}_liteparse"
        parse_dir = doc_dir / "parse"
        parse_dir.mkdir(parents=True, exist_ok=True)

        result = self._parser.parse(str(source))
        md_text = _clean_markdown(result.text or "")

        # Validate BEFORE writing, so failed parses leave no orphan files behind
        if len(md_text) < PARSER_MIN_CONTENT_LENGTH:
            raise ValueError(
                f"Parsed content is suspiciously short ({len(md_text)} chars). "
                f"'{source.name}' may be a scanned PDF — enable OCR "
                f"(PARSER_ENABLE_OCR=true)."
            )

        # converted.md — human-readable audit artifact; chunking input
        out_file = parse_dir / "converted.md"
        out_file.write_text(md_text, encoding="utf-8")
        print(f"[parser:liteparse] Saved markdown: {out_file} ({result.num_pages} pages)")

        # liteparse_pages.json — page-level font metadata; mirrors docling.json role
        pages = getattr(result, "pages", None) or []
        pages_data = [_page_to_dict(p) for p in pages]
        (parse_dir / "liteparse_pages.json").write_text(
            json.dumps(pages_data, indent=2), encoding="utf-8"
        )
        print(f"[parser:liteparse] Saved page metadata: {parse_dir / 'liteparse_pages.json'} ({len(pages)} pages)")

        # parse_result.json — ParseResult metadata; ties all parse artifacts together
        parse_result_meta = {
            "content_hash": content_hash,
            "source_path": str(source),
            "content_type": source.suffix.lower().lstrip("."),
        }
        (parse_dir / "parse_result.json").write_text(
            json.dumps(parse_result_meta, indent=2), encoding="utf-8"
        )
        print(f"[parser:liteparse] Saved parse result: {parse_dir / 'parse_result.json'}")

        return ParseResult(
            markdown=md_text,
            content_hash=content_hash,
            source_path=str(source),
            content_type=source.suffix.lower().lstrip("."),
            parser="liteparse",
            liteparse_pages=pages if pages else None,
            doc_dir=doc_dir,
        )
