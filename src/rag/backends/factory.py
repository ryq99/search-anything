from dataclasses import dataclass

from rag.config import (
    CLOUD_BACKEND,
    LOCAL_SUMMARY_MODEL, CLOUD_SUMMARY_MODEL,
    LOCAL_SYNTHESIS_MODEL, CLOUD_SYNTHESIS_MODEL,
)


@dataclass
class Backend:
    registry: object
    vectorstore: object
    summary_llm: object    # used at indexing time for chunk summarization
    synthesis_llm: object  # used at query time for answer generation


def get_backend() -> Backend:
    """Return the concrete backend selected by CLOUD_BACKEND env var."""
    if CLOUD_BACKEND == "aws":
        from rag.backends.aws.registry import DynamoDBRegistry
        from rag.backends.aws.vectorstore import BedrockKBVectorStore
        from rag.backends.aws.llm import BedrockLLM
        return Backend(
            registry=DynamoDBRegistry(),
            vectorstore=BedrockKBVectorStore(),
            summary_llm=BedrockLLM(CLOUD_SUMMARY_MODEL),
            synthesis_llm=BedrockLLM(CLOUD_SYNTHESIS_MODEL),
        )

    from rag.backends.local.registry import JsonRegistry
    from rag.backends.local.vectorstore import MilvusVectorStore
    from rag.backends.local.llm import LocalLLM
    return Backend(
        registry=JsonRegistry(),
        vectorstore=MilvusVectorStore(),
        summary_llm=LocalLLM(LOCAL_SUMMARY_MODEL),
        synthesis_llm=LocalLLM(LOCAL_SYNTHESIS_MODEL),
    )
