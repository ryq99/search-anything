import asyncio
import time

import boto3

from rag.config import BEDROCK_REGION


class BedrockLLM:
    """LLM backend using AWS Bedrock converse API.

    Supports any model available via Bedrock cross-region inference
    (e.g. us.anthropic.claude-haiku-4-5-20251001). Auth via IAM — no API key needed.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        for attempt, wait in enumerate([0, 60, 120, 240, 480]):
            try:
                if wait:
                    time.sleep(wait)
                resp = self._client.converse(
                    modelId=self.model,
                    system=[{"text": system}],
                    messages=[{"role": "user", "content": [{"text": user}]}],
                    inferenceConfig={"maxTokens": max_tokens},
                )
                return resp["output"]["message"]["content"][0]["text"]
            except self._client.exceptions.ThrottlingException:
                if attempt == 4:
                    raise

    async def acomplete(self, system: str, user: str, max_tokens: int) -> str:
        # boto3 is sync-only; offload to thread pool so concurrent summarize_chunks()
        # calls don't block the event loop
        return await asyncio.to_thread(self.complete, system, user, max_tokens)
