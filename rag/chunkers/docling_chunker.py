from pathlib import Path

from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from rag.config import EMBED_MODEL_ID, CHUNK_MAX_TOKENS


class DoclingChunker:
    """
    Chunks a pre-converted markdown file using docling's HybridChunker via
    DoclingLoader.

    HybridChunker is a token-aware splitter that respects the document's
    structural boundaries (headings, tables, lists) rather than slicing on
    raw character offsets. It also records, for every chunk, the ordered
    heading stack that contains it in dl_meta["headings"] — outermost to
    innermost — which we use to derive parent_headings for section-level
    summarization without a separate TOC extraction step.
    """

    def chunk(self, md_path: Path, content_hash: str, filename: str) -> tuple[list[Document], dict[str, str]]:
        """
        Chunk the markdown at md_path and enrich each Document with heading
        ancestry metadata derived from docling's dl_meta.

        Returns:
            splits: list of enriched Documents ready for embedding
            parent_headings_text: {parent_heading_path: concatenated_chunk_text}
                                  keyed by parent heading path (for summarization)
        """
        loader = DoclingLoader(
            file_path=[str(md_path)],
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(
                tokenizer=EMBED_MODEL_ID,
                max_tokens=CHUNK_MAX_TOKENS,
            ),
        )
        docs = loader.load()

        splits: list[Document] = []
        parent_headings_text: dict[str, str] = {}

        for doc in docs:
            dl_meta = doc.metadata.get("dl_meta", {})
            headings_list = dl_meta.get("headings", [])

            # Ancestry from the heading stack: everything above the innermost
            # heading is the parent path.
            headings_str = " => ".join(headings_list)
            parent_headings_str = " => ".join(headings_list[:-1]) if len(headings_list) > 1 else ""

            splits.append(Document(
                page_content=doc.page_content,
                metadata={
                    "content_hash": content_hash,
                    "filename": filename,
                    "headings": headings_str,
                    "parent_headings": parent_headings_str,
                },
            ))

            if parent_headings_str:
                if parent_headings_str not in parent_headings_text:
                    parent_headings_text[parent_headings_str] = doc.page_content
                else:
                    parent_headings_text[parent_headings_str] += "\n" + doc.page_content

        return splits, parent_headings_text
