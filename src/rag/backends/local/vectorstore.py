from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from rag.core.schemas import Chunk
from rag.config import EMBED_MODEL_ID, VECTOR_STORE_NAME, MILVUS_URI, VECTOR_STORE_DIR


class MilvusVectorStore:
    """Vector store backend using local Milvus (SQLite-based .db file)."""

    def __init__(self) -> None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._embedding = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs={"trust_remote_code": True},
        )

    def store(self, chunks: list[Chunk]) -> None:
        """Embed and persist chunks. Converts Chunk -> langchain Document at this boundary."""
        docs = self._to_documents(chunks)
        if not Path(MILVUS_URI).exists():
            print("[vectorstore] Creating new Milvus collection...")
            Milvus.from_documents(
                documents=docs,
                embedding=self._embedding,
                collection_name=VECTOR_STORE_NAME,
                connection_args={"uri": MILVUS_URI},
                index_params={"index_type": "FLAT"},
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
            index_params={"index_type": "FLAT"},
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
                    "text": chunk.text,  # raw text preserved for display
                },
            ))
        return docs
