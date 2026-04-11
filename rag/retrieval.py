from pathlib import Path

import pandas as pd
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from rag.config import (
    EMBED_MODEL_ID, VECTOR_STORE_NAME, MILVUS_URI,
    RETRIEVAL_K, RETRIEVAL_EXPR,
)


def _get_vectorstore() -> Milvus:
    """Lazy-init Milvus connection. Call once and reuse for Lambda warm paths."""
    embedding = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"trust_remote_code": True},
    )
    return Milvus(
        embedding_function=embedding,
        collection_name=VECTOR_STORE_NAME,
        connection_args={"uri": MILVUS_URI},
        index_params={"index_type": "FLAT"},
    )


def _load_summaries_for_books(registry: dict) -> pd.DataFrame:
    """Concatenate all per-book summary CSVs into one DataFrame."""
    frames = []
    for entry in registry.get("books", {}).values():
        csv_path = entry.get("summary_csv", "")
        if csv_path and Path(csv_path).exists():
            frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame(columns=["Parent Headings", "Summary"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["Parent Headings"], keep="last"
    )


def retrieve(
    query: str,
    vectorstore: Milvus | None = None,
    summaries_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Run similarity search and enrich each result with its pre-computed summary.
    Returns a list of dicts with rank, headings, parent_headings, child_headings,
    summary, and page_content.
    """
    if vectorstore is None:
        vectorstore = _get_vectorstore()

    docs = vectorstore.similarity_search(
        query,
        k=RETRIEVAL_K,
        expr=RETRIEVAL_EXPR,
    )

    chunks = []
    for i, doc in enumerate(docs):
        parent_headings = doc.metadata.get("parent_headings", "")
        summary = "Not Available"

        if summaries_df is not None and not summaries_df.empty and parent_headings:
            matches = summaries_df.loc[
                summaries_df["Parent Headings"] == parent_headings, "Summary"
            ].values.tolist()
            if matches:
                summary = matches[0]

        chunks.append({
            "rank": i + 1,
            "headings": doc.metadata.get("headings", ""),
            "parent_headings": parent_headings,
            "child_headings": doc.metadata.get("child_headings", ""),
            "summary": summary,
            "page_content": doc.page_content,
        })

    return chunks


def format_retrieval_results(chunks: list[dict]) -> str:
    """Format structured chunks into the exact string used in the synthesis prompt."""
    parts = []
    for c in chunks:
        parts.append(
            f"====== Retrieved Chunk {c['rank']} ======\n\n"
            f"Parent Headings:\n    {c['parent_headings']}\n"
            f"Summary:\n    {c['summary']}\n"
            f"Page Content:\n    {c['page_content']}\n"
        )
    return "\n".join(parts)
