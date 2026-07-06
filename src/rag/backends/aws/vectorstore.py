import json
import time

import boto3
from langchain_core.documents import Document

from rag.core.schemas import Chunk
from rag.config import (
    AWS_REGION, BEDROCK_REGION,
    S3_BUCKET, BEDROCK_KNOWLEDGE_BASE_ID, BEDROCK_DATA_SOURCE_ID,
    PIPELINE_CONFIG_HASH,
    KB_SYNC_POLL_INTERVAL, KB_SYNC_TIMEOUT,
)


class BedrockKBVectorStore:
    """Vector store backend using Bedrock Knowledge Bases with chunkingStrategy=NONE.

    Ingestion uploads pre-chunked .txt + .txt.metadata.json sidecar files to S3
    under chunks/{pipeline_config_hash}/, then triggers a KB ingestion job.
    Retrieval calls the bedrock-agent-runtime Retrieve API.

    Pre-requisites (provisioned outside this code):
      - S3 bucket with chunks/ prefix
      - Bedrock KB with S3 data source, chunkingStrategy=NONE
    """

    def __init__(self) -> None:
        self._s3      = boto3.client("s3", region_name=AWS_REGION)
        self._agent   = boto3.client("bedrock-agent", region_name=BEDROCK_REGION)
        self._runtime = boto3.client("bedrock-agent-runtime", region_name=BEDROCK_REGION)

    def store(self, chunks: list[Chunk]) -> None:
        for i, chunk in enumerate(chunks):
            key = f"chunks/{PIPELINE_CONFIG_HASH}/{chunk.content_hash}_{i}.txt"
            self._s3.put_object(Bucket=S3_BUCKET, Key=key, Body=chunk.enriched_text.encode())
            meta = {"metadataAttributes": {
                "raw_text":            chunk.text,
                "headings":            chunk.headings,
                "parent_headings":     chunk.parent_headings,
                "summary":             chunk.summary,
                "filename":            chunk.filename,
                "pipeline_config_hash": PIPELINE_CONFIG_HASH,
            }}
            self._s3.put_object(
                Bucket=S3_BUCKET,
                Key=key + ".metadata.json",
                Body=json.dumps(meta).encode(),
            )
        job = self._agent.start_ingestion_job(
            knowledgeBaseId=BEDROCK_KNOWLEDGE_BASE_ID,
            dataSourceId=BEDROCK_DATA_SOURCE_ID,
        )
        job_id = job["ingestionJob"]["ingestionJobId"]
        print(f"[vectorstore] KB ingestion job {job_id} started — waiting for sync...")
        self._wait_for_sync(job_id)
        print(f"[vectorstore] KB sync complete — {len(chunks)} chunks searchable.")

    def _wait_for_sync(self, job_id: str) -> None:
        """Block until the KB ingestion job reaches a terminal state.

        Raises on FAILED or timeout so the caller never registers a false success —
        build() runs store() before registry.register(), so a raise here leaves the
        document un-ingested for the next run.
        """
        deadline = time.time() + KB_SYNC_TIMEOUT
        while time.time() < deadline:
            resp = self._agent.get_ingestion_job(
                knowledgeBaseId=BEDROCK_KNOWLEDGE_BASE_ID,
                dataSourceId=BEDROCK_DATA_SOURCE_ID,
                ingestionJobId=job_id,
            )
            status = resp["ingestionJob"]["status"]
            if status == "COMPLETE":
                return
            if status == "FAILED":
                reasons = resp["ingestionJob"].get("failureReasons", [])
                raise RuntimeError(f"Bedrock KB ingestion job {job_id} FAILED: {reasons}")
            print(f"[vectorstore] KB sync status: {status}")
            time.sleep(KB_SYNC_POLL_INTERVAL)
        raise TimeoutError(f"KB ingestion job {job_id} did not complete within {KB_SYNC_TIMEOUT}s")

    def get_store(self) -> "_BedrockKBStore":
        return _BedrockKBStore(self._runtime)


class _BedrockKBStore:
    """Thin wrapper exposing similarity_search() over the Bedrock KB Retrieve API."""

    def __init__(self, runtime) -> None:
        self._runtime = runtime

    def similarity_search(self, query: str, k: int, expr: str = None) -> list[Document]:
        # expr is Milvus-style syntax — not supported by Bedrock KB; ignored here.
        # Use BEDROCK_RETRIEVAL_FILTER env var for AWS-native metadata filtering if needed.
        resp = self._runtime.retrieve(
            knowledgeBaseId=BEDROCK_KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": k},
            },
        )
        return [
            Document(
                page_content=r["content"]["text"],
                metadata={
                    "raw_text":        r["metadata"].get("raw_text", ""),
                    "headings":        r["metadata"].get("headings", ""),
                    "parent_headings": r["metadata"].get("parent_headings", ""),
                    "summary":         r["metadata"].get("summary", ""),
                    "filename":        r["metadata"].get("filename", ""),
                },
            )
            for r in resp["retrievalResults"]
        ]
