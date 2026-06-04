from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from rag.config import EMBED_MODEL_ID, VECTOR_STORE_NAME, MILVUS_URI, VECTOR_STORE_DIR


class MilvusVectorStore:
    """Vector store backend using local Milvus (SQLite-based .db file)."""

    def __init__(self) -> None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._embedding = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs={"trust_remote_code": True},
        )

    def store(self, docs: list[Document]) -> None:
        """Embed and persist documents. Converts content_hash to int64 for Milvus schema."""
        adapted = self._adapt_metadata(docs)
        if not Path(MILVUS_URI).exists():
            print("[vectorstore] Creating new Milvus collection...")
            Milvus.from_documents(
                documents=adapted,
                embedding=self._embedding,
                collection_name=VECTOR_STORE_NAME,
                connection_args={"uri": MILVUS_URI},
                index_params={"index_type": "FLAT"},
            )
        else:
            print("[vectorstore] Adding to existing Milvus collection...")
            vs = self._connect()
            vs.add_documents(adapted)

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
    def _adapt_metadata(docs: list[Document]) -> list[Document]:
        # Milvus schema uses int64 for the hash field; convert hex string here
        adapted = []
        for doc in docs:
            meta = dict(doc.metadata)
            hash_str = meta.pop("content_hash", "0x0")
            meta["binary_hash"] = int(hash_str, 16) % (2 ** 63)
            adapted.append(Document(page_content=doc.page_content, metadata=meta))
        return adapted
