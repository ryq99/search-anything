from pathlib import Path

import pandas as pd

from rag.config import RETRIEVAL_K, RETRIEVAL_EXPR


def _get_vectorstore():
    from rag.backends.factory import get_backend
    return get_backend().vectorstore.get_store()


def _load_summaries_for_books(registry: dict) -> pd.DataFrame:
    """Concatenate all per-book summary CSVs into one DataFrame."""
    frames = []
    for entry in registry.get("books", {}).values():
        # support both old key (summary_csv) and new key (summary_artifact_path)
        csv_path = entry.get("summary_artifact_path") or entry.get("summary_csv", "")
        if csv_path and Path(csv_path).exists():
            frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame(columns=["Parent Headings", "Summary"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["Parent Headings"], keep="last"
    )


def retrieve(
    query: str,
    vectorstore=None,
    summaries_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Run similarity search and enrich each result with its pre-computed summary.
    Returns a list of dicts with rank, headings, parent_headings,
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
            "summary": summary,
            "page_content": doc.page_content,
        })

    return chunks


def format_retrieval_results(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"====== Retrieved Chunk {c['rank']} ======\n\n"
            f"Parent Headings:\n    {c['parent_headings']}\n"
            f"Summary:\n    {c['summary']}\n"
            f"Page Content:\n    {c['page_content']}\n"
        )
    return "\n".join(parts)
