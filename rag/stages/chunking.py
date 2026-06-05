from pathlib import Path

from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from rag.config import EMBED_MODEL_ID, CHUNK_MAX_TOKENS


def chunk_and_enrich(
    md_path: Path,
    content_hash: str,
    filename: str,
) -> tuple[list[Document], dict[str, str]]:
    """
    Load markdown with DoclingLoader + HybridChunker and enrich each chunk with
    its heading ancestry, derived directly from docling's chunk metadata.

    docling's HybridChunker records, for every chunk, the ordered list of headings
    that contain it (outermost -> innermost) in dl_meta["headings"]. That list IS
    the parent -> child path, so we read ancestry straight from the document
    structure instead of joining against a separately-extracted TOC:

        headings = ["1 Introduction", "1.2 Motivation"]
                    └── parent ──────┘ └── this chunk ─┘

    content_hash is stored as a plain string in Document metadata — backend
    implementations are responsible for any type adaptation (e.g. int64 for Milvus).

    Returns (chunks, parent_headings_text) where parent_headings_text maps each
    parent-heading path to the concatenated text of all chunks beneath it, ready
    for section-level summarization.
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

        # Ancestry comes straight from the heading stack: everything above the
        # innermost heading is the parent path.
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

    print(f"[chunking] Created {len(splits)} chunks, {len(parent_headings_text)} parent heading groups")
    return splits, parent_headings_text
