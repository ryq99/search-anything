class BedrockLLM:
    """LLM backend using AWS Bedrock converse API."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "AWS backend not yet wired. "
            "Set CLOUD_BACKEND=local or implement BedrockLLM using boto3 bedrock-runtime converse()."
        )

    def complete(self, system: str, user: str, max_tokens: int) -> str: ...
    async def acomplete(self, system: str, user: str, max_tokens: int) -> str: ...
