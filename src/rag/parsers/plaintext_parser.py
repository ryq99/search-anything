import hashlib
import json
from pathlib import Path

from rag.core.schemas import ParseResult
from rag.config import DATA_DIR


class PlaintextParser:
    """Passthrough parser for .md and .txt files. No conversion — reads as-is."""

    def parse(self, source: Path | str) -> ParseResult:
        source = Path(source)
        raw = source.read_bytes()
        content_hash = hashlib.sha256(raw).hexdigest()
        content_type = source.suffix.lower().lstrip(".")
        stem = source.stem.replace(" ", "_").replace("-", "_").lower()
        parse_dir = DATA_DIR / f"{stem}_{content_hash[:12]}_plaintext" / "parse"
        parse_dir.mkdir(parents=True, exist_ok=True)
        (parse_dir / "converted.md").write_bytes(raw)
        (parse_dir / "parse_result.json").write_text(
            json.dumps({"content_hash": content_hash, "source_path": str(source), "content_type": content_type}, indent=2)
        )
        return ParseResult(
            markdown=raw.decode("utf-8"),
            content_hash=content_hash,
            source_path=str(source),
            content_type=content_type,
            parser="plaintext",
            doc_dir=parse_dir.parent,
        )
