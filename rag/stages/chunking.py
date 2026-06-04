from pathlib import Path

import pandas as pd
from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from rag.config import EMBED_MODEL_ID, CHUNK_MAX_TOKENS


def chunk_and_enrich(
    md_path: Path,
    toc_df: pd.DataFrame,
    content_hash: str,
    filename: str,
) -> tuple[list[Document], dict[str, str]]:
    """
    Load markdown with DoclingLoader + HybridChunker, enrich each chunk with
    parent/child heading metadata from the TOC.

    content_hash is stored as a plain string in Document metadata — backend
    implementations are responsible for any type adaptation (e.g. int64 for Milvus).
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

        if not headings_list:
            splits.append(Document(
                page_content=doc.page_content,
                metadata={
                    "content_hash": content_hash,
                    "filename": filename,
                    "headings": "",
                    "parent_headings": "",
                    "child_headings": "",
                },
            ))
            continue

        heading = headings_list[0]
        nth_level = len(heading.split(" ")[0].split("."))

        parent_headings_val: list[str] = []
        child_headings_val: list[str] = []

        if not toc_df.empty and nth_level <= toc_df.shape[1]:
            level_col = f"Level {nth_level}"
            if level_col in toc_df.columns:
                matched = toc_df.loc[toc_df[level_col].str.contains(heading, regex=False)]
                if not matched.empty:
                    parent_cols = [f"Level {i + 1}" for i in range(nth_level - 1)]
                    if parent_cols:
                        parent_headings_val = sorted(
                            matched[parent_cols].drop_duplicates().values.reshape(-1).tolist()
                        )
                    child_cols = [
                        f"Level {i + 1}"
                        for i in range(nth_level, toc_df.shape[1])
                        if f"Level {i + 1}" in toc_df.columns
                    ]
                    if child_cols:
                        child_headings_val = sorted(
                            matched[child_cols].drop_duplicates().values.reshape(-1).tolist()
                        )

        parent_headings_str = " => ".join(h for h in parent_headings_val if h)
        child_headings_str = " => ".join(h for h in child_headings_val if h)
        headings_str = " => ".join(headings_list)

        splits.append(Document(
            page_content=doc.page_content,
            metadata={
                "content_hash": content_hash,
                "filename": filename,
                "headings": headings_str,
                "parent_headings": parent_headings_str,
                "child_headings": child_headings_str,
            },
        ))

        if parent_headings_str:
            if parent_headings_str not in parent_headings_text:
                parent_headings_text[parent_headings_str] = doc.page_content
            else:
                parent_headings_text[parent_headings_str] += "\n" + doc.page_content

    print(f"[chunking] Created {len(splits)} chunks, {len(parent_headings_text)} parent heading groups")
    return splits, parent_headings_text
