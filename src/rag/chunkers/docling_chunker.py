import re
from pathlib import Path

from docling.chunking import HybridChunker

from rag.core.schemas import Chunk, ParseResult
from rag.config import (
    CHUNK_TOKENIZER,
    CHUNK_MAX_TOKENS,
    CHUNK_MERGE_PEERS,
    CHUNK_MERGE_LIST_ITEMS,
)


class _HeadingHierarchy:
    """
    Reconstructs heading ancestry across sequential chunks.

    Docling returns only the immediate heading per chunk, not the full ancestor
    chain. This class maintains a level-indexed state and infers ancestry from
    section number prefixes (e.g. "1.2.1" → level 3, ancestors "1", "1.2").

    Eviction rules:
      - Numbered entries that are not numeric ancestors of the current heading
        are evicted.
      - Non-numbered entries (e.g. "Preface") are evicted when a numbered
        heading is pushed — they are frontmatter, not structural ancestors.
    """

    def __init__(self) -> None:
        self._levels: dict[int, str] = {}

    def update(self, heading: str) -> None:
        level = _heading_level(heading)
        for l in [l for l in self._levels if l >= level]:
            del self._levels[l]
        ancestors = _ancestor_numbers(heading)
        is_numbered = bool(_section_number(heading))
        for l in list(self._levels):
            entry_num = _section_number(self._levels[l])
            if (entry_num and entry_num not in ancestors) or (not entry_num and is_numbered):
                del self._levels[l]
        self._levels[level] = heading

    @property
    def path(self) -> str:
        return " => ".join(self._levels[l] for l in sorted(self._levels))

    @property
    def parent_path(self) -> str:
        levels = sorted(self._levels)
        return " => ".join(self._levels[l] for l in levels[:-1])


def _section_number(heading: str) -> str:
    token = heading.split()[0] if heading.split() else ""
    return token if re.match(r"^\d+(\.\d+)*$", token) else ""


def _heading_level(heading: str) -> int:
    num = _section_number(heading)
    return num.count(".") + 1 if num else 1


def _ancestor_numbers(heading: str) -> set[str]:
    num = _section_number(heading)
    if not num:
        return set()
    parts = num.split(".")
    return {".".join(parts[:i + 1]) for i in range(len(parts) - 1)}


class DoclingChunker:
    """
    Chunks a document using docling's native HybridChunker.

    When the ParseResult carries a live DoclingDocument (set by DoclingParser),
    the chunker operates on the original rich document structure — tables, heading
    levels, reading order — without any markdown round-trip.

    When DoclingDocument is absent (liteparse or plaintext path), it falls back
    to loading the persisted docling.json, then to re-parsing converted.md via
    docling's markdown backend (lower structural fidelity; last resort).

    Heading ancestry is reconstructed via _HeadingHierarchy since docling's
    meta.headings returns only the immediate heading, not the full ancestor chain.

    Text stored in enriched_text is the heading-prefixed version produced by
    chunker.contextualize() — this is what gets embedded, as heading context
    materially improves retrieval quality for section-level queries.
    """

    def __init__(self) -> None:
        self._chunker = HybridChunker(
            tokenizer=CHUNK_TOKENIZER,
            max_tokens=CHUNK_MAX_TOKENS,
            merge_peers=CHUNK_MERGE_PEERS,
        )
        inner = getattr(self._chunker, "_inner_chunker", None)
        if inner is not None and hasattr(inner, "merge_list_items"):
            try:
                inner.merge_list_items = CHUNK_MERGE_LIST_ITEMS
            except (AttributeError, ValueError):
                pass

    def chunk(self, parse_result: ParseResult) -> tuple[list[Chunk], dict[str, str]]:
        dl_doc = self._get_docling_document(parse_result)
        hierarchy = _HeadingHierarchy()
        chunks: list[Chunk] = []
        parent_headings_text: dict[str, str] = {}

        for dl_chunk in self._chunker.chunk(dl_doc=dl_doc):
            raw_heading = (dl_chunk.meta.headings or [""])[0]
            if raw_heading:
                hierarchy.update(raw_heading)

            chunk = Chunk(
                text=dl_chunk.text,
                enriched_text=self._chunker.contextualize(chunk=dl_chunk),
                headings=hierarchy.path,
                parent_headings=hierarchy.parent_path,
                content_hash=parse_result.content_hash,
                filename=parse_result.source_path.rsplit("/", 1)[-1],
            )
            chunks.append(chunk)

            if hierarchy.parent_path:
                parent_headings_text[hierarchy.parent_path] = (
                    parent_headings_text.get(hierarchy.parent_path, "") + "\n" + dl_chunk.text
                ).lstrip("\n")

        return chunks, parent_headings_text

    def _get_docling_document(self, parse_result: ParseResult):
        """
        Return the DoclingDocument to chunk.

        Priority:
          1. Live in-memory object from DoclingParser (no I/O).
          2. Persisted docling.json — enables re-chunking without re-parsing.
          3. Markdown fallback — re-parse converted.md via docling's markdown
             backend (lower structural fidelity; last resort).
        """
        if parse_result.docling_document is not None:
            return parse_result.docling_document

        if parse_result.doc_dir is not None:
            docling_json = parse_result.doc_dir / "parse" / "docling.json"
            if docling_json.exists():
                import json
                from docling_core.types.doc import DoclingDocument
                data = json.loads(docling_json.read_text(encoding="utf-8"))
                return DoclingDocument.model_validate(data)

        from docling.document_converter import DocumentConverter
        if parse_result.doc_dir is not None:
            md_path = parse_result.doc_dir / "parse" / "converted.md"
        else:
            from rag.config import DATA_DIR
            stem = Path(parse_result.source_path).stem.replace(" ", "_").replace("-", "_").lower()
            md_path = DATA_DIR / f"{stem}_converted.md"
        return DocumentConverter().convert(str(md_path)).document
