from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from rag.core.schemas import Chunk
from rag.config import (
    EMBED_MODEL_ID, VECTOR_STORE_NAME, MILVUS_URI, VECTOR_STORE_DIR,
    MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE,
    MILVUS_HNSW_M, MILVUS_HNSW_EF_CONSTRUCTION,
    MILVUS_IVF_NLIST,
)


def _build_index_params() -> dict:
    params: dict = {"index_type": MILVUS_INDEX_TYPE, "metric_type": MILVUS_METRIC_TYPE}
    if MILVUS_INDEX_TYPE == "HNSW":
        params["params"] = {"M": MILVUS_HNSW_M, "efConstruction": MILVUS_HNSW_EF_CONSTRUCTION}
    elif MILVUS_INDEX_TYPE.startswith("IVF"):
        params["params"] = {"nlist": MILVUS_IVF_NLIST}
    return params


class MilvusVectorStore:
    """Vector store backend using local Milvus (SQLite-based .db file)."""

    def __init__(self) -> None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._embedding = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs={"trust_remote_code": True},
        )

    def store(self, chunks: list[Chunk], superseded: list[str] | None = None) -> None:
        """Embed and persist chunks. Converts Chunk -> langchain Document at this boundary.

        Older versions of the same document (`superseded` content_hashes) are deleted
        first, matched by the same binary_hash used in _to_documents, so a query never
        sees the old and new version at once.
        """
        if superseded and Path(MILVUS_URI).exists():
            store = self._connect()
            for old_content_hash in superseded:
                binary_hash = int(old_content_hash, 16) % (2 ** 63)
                store.delete(expr=f"binary_hash == {binary_hash}")
                print(f"[vectorstore] Superseded {old_content_hash[:10]}... — removed old vectors.")

        docs = self._to_documents(chunks)
        if not Path(MILVUS_URI).exists():
            print("[vectorstore] Creating new Milvus collection...")
            Milvus.from_documents(
                documents=docs,
                embedding=self._embedding,
                collection_name=VECTOR_STORE_NAME,
                connection_args={"uri": MILVUS_URI},
                index_params=_build_index_params(),
            )
        else:
            print("[vectorstore] Adding to existing Milvus collection...")
            self._connect().add_documents(docs)

    def get_store(self) -> Milvus:
        return self._connect()

    def _connect(self) -> Milvus:
        return Milvus(
            embedding_function=self._embedding,
            collection_name=VECTOR_STORE_NAME,
            connection_args={"uri": MILVUS_URI},
            index_params=_build_index_params(),
        )

    @staticmethod
    def _to_documents(chunks: list[Chunk]) -> list[Document]:
        """
        Convert Chunk objects to langchain Documents for Milvus.

        - page_content uses enriched_text (heading-prefixed) so the embedding
          captures section context, improving retrieval on section-level queries.
        - content_hash is converted to int64 (binary_hash) to satisfy Milvus's
          schema requirement; the original hex string is preserved in metadata.
        - headings and parent_headings are stored as strings for metadata filtering.
        """
        docs = []
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk.enriched_text,
                metadata={
                    "binary_hash": int(chunk.content_hash, 16) % (2 ** 63),
                    "filename": chunk.filename,
                    "headings": chunk.headings,
                    "parent_headings": chunk.parent_headings,
                    "summary": chunk.summary,
                    "raw_text": chunk.text,  # raw text preserved for display
                },
            ))
        return docs
