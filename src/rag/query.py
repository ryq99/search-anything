"""Read path: turn a question into a synthesized answer.

Counterpart to indexing.py (the write path). Orchestrates retrieval over the
active backend's vector store and hands the results to synthesis.
"""
from rag.retrieval import retrieve, format_retrieval_results, _load_summaries_for_books
from rag.synthesis import ask as _synthesize
from rag.backends.factory import get_backend


def ask(question: str) -> str:
    """Ask any ML question. Retrieves relevant chunks and synthesizes an answer."""
    backend = get_backend()
    registry = backend.registry.load_all()
    summaries_df = _load_summaries_for_books(registry)
    vs = backend.vectorstore.get_store()
    chunks = retrieve(question, vectorstore=vs, summaries_df=summaries_df)
    retrieval_str = format_retrieval_results(chunks)
    return _synthesize(question, retrieval_str)
