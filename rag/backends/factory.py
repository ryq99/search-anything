from dataclasses import dataclass

from rag.config import CLOUD_BACKEND


@dataclass
class Backend:
    registry: object
    vectorstore: object
    llm: object


def get_backend() -> Backend:
    """Return the concrete backend set selected by CLOUD_BACKEND env var."""
    if CLOUD_BACKEND == "aws":
        from rag.backends.aws.registry import DynamoDBRegistry
        from rag.backends.aws.vectorstore import BedrockKBVectorStore
        from rag.backends.aws.llm import BedrockLLM
        return Backend(
            registry=DynamoDBRegistry(),
            vectorstore=BedrockKBVectorStore(),
            llm=BedrockLLM(),
        )

    from rag.backends.local.registry import JsonRegistry
    from rag.backends.local.vectorstore import MilvusVectorStore
    from rag.backends.local.llm import AnthropicLLM
    return Backend(
        registry=JsonRegistry(),
        vectorstore=MilvusVectorStore(),
        llm=AnthropicLLM(),
    )
