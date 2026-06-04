class BedrockKBVectorStore:
    """
    Vector store backend using Bedrock Knowledge Bases with chunkingStrategy=NONE.

    Ingestion writes pre-chunked .txt + .txt.metadata.json sidecar files to S3.
    Retrieval calls bedrock-agent-runtime Retrieve API with metadata filters.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "AWS backend not yet wired. "
            "Set CLOUD_BACKEND=local or implement BedrockKBVectorStore using boto3."
        )

    def store(self, docs) -> None: ...
    def get_store(self): ...
