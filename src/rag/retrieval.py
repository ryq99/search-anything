from rag.config import RETRIEVAL_K, RETRIEVAL_EXPR
from rag.synthesis import ask as _synthesize


def _get_vectorstore():
    from rag.backends.factory import get_backend
    return get_backend().vectorstore.get_store()


def retrieve(query: str, vectorstore=None) -> list[dict]:
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
        chunks.append({
            "rank": i + 1,
            "headings": doc.metadata.get("headings", ""),
            "parent_headings": doc.metadata.get("parent_headings", ""),
            "summary": doc.metadata.get("summary", ""),
            "page_content": doc.metadata.get("raw_text") or doc.page_content,
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


def ask(question: str) -> str:
    """Retrieve relevant chunks and synthesize an answer."""
    from rag.backends.factory import get_backend
    backend = get_backend()
    chunks = retrieve(question, vectorstore=backend.vectorstore.get_store())
    return _synthesize(question, format_retrieval_results(chunks), backend.synthesis_llm)
