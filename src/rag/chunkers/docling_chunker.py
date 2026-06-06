from pathlib import Path

from docling.chunking import HybridChunker

from rag.core.schemas import Chunk, ParseResult
from rag.config import (
    CHUNK_TOKENIZER,
    CHUNK_MAX_TOKENS,
    CHUNK_MERGE_PEERS,
    CHUNK_MERGE_LIST_ITEMS,
)


class DoclingChunker:
    """
    Chunks a document using docling's native HybridChunker.

    When the ParseResult carries a live DoclingDocument (set by DoclingParser),
    the chunker operates on the original rich document structure — tables, heading
    levels, reading order — without any markdown round-trip.

    When DoclingDocument is absent (liteparse or plaintext path), it falls back
    to re-parsing the saved markdown file via docling's markdown backend. The
    chunking logic is identical; only the structural fidelity differs.

    Text stored in enriched_text is the heading-prefixed version produced by
    chunker.contextualize() — this is what gets embedded, as heading context
    materially improves retrieval quality for section-level queries.
    """

    def __init__(self) -> None:
        # HybridChunker runs a 3-pass pipeline:
        #   1. HierarchicalChunker  — split on document structure (headings, lists)
        #   2. token-aware split    — break any chunk exceeding max_tokens
        #   3. merge_peers          — recombine undersized same-heading neighbours
        # All knobs are sourced from config so they can be tuned via .env.
        self._chunker = HybridChunker(
            tokenizer=CHUNK_TOKENIZER,
            max_tokens=CHUNK_MAX_TOKENS,
            merge_peers=CHUNK_MERGE_PEERS,
        )
        # merge_list_items belongs to the inner hierarchical (1st) pass, which
        # HybridChunker constructs internally. Apply it there when supported so
        # the behaviour stays configurable across docling versions.
        inner = getattr(self._chunker, "_inner_chunker", None)
        if inner is not None and hasattr(inner, "merge_list_items"):
            try:
                inner.merge_list_items = CHUNK_MERGE_LIST_ITEMS
            except (AttributeError, ValueError):
                pass  # frozen model on some versions; default (True) stands

    def chunk(self, parse_result: ParseResult) -> tuple[list[Chunk], dict[str, str]]:
        """
        Chunk the document and return enriched Chunk objects.

        Returns:
            chunks: list of Chunk ready for embedding and storage
            parent_headings_text: {parent_heading_path: concatenated raw text}
                                  used by the summarization stage
        """
        dl_doc = self._get_docling_document(parse_result)

        chunks: list[Chunk] = []
        parent_headings_text: dict[str, str] = {}

        for dl_chunk in self._chunker.chunk(dl_doc=dl_doc):
            headings_list: list[str] = dl_chunk.meta.headings or []
            headings_str = " => ".join(headings_list)
            parent_headings_str = " => ".join(headings_list[:-1]) if len(headings_list) > 1 else ""

            enriched = self._chunker.contextualize(chunk=dl_chunk)

            chunk = Chunk(
                text=dl_chunk.text,
                enriched_text=enriched,
                headings=headings_str,
                parent_headings=parent_headings_str,
                content_hash=parse_result.content_hash,
                filename=parse_result.source_path.rsplit("/", 1)[-1],
            )
            chunks.append(chunk)

            if parent_headings_str:
                if parent_headings_str not in parent_headings_text:
                    parent_headings_text[parent_headings_str] = dl_chunk.text
                else:
                    parent_headings_text[parent_headings_str] += "\n" + dl_chunk.text

        return chunks, parent_headings_text

    def _get_docling_document(self, parse_result: ParseResult):
        """
        Return the DoclingDocument to chunk.

        Prefers the live object from DoclingParser (no round-trip). Falls back
        to re-parsing the markdown artifact when DoclingDocument is unavailable
        (liteparse or plaintext paths).
        """
        if parse_result.docling_document is not None:
            return parse_result.docling_document

        # Markdown fallback: re-parse via docling's markdown backend
        from docling.document_converter import DocumentConverter
        from rag.config import DATA_DIR
        stem = Path(parse_result.source_path).stem.replace(" ", "_").replace("-", "_").lower()
        md_path = DATA_DIR / f"{stem}_converted.md"
        return DocumentConverter().convert(str(md_path)).document
