import hashlib
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
    RETRIEVAL_EXCLUDE_HEADINGS,
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

    def store(self, chunks: list[Chunk], superseded: list[str] | None = None) -> None:
        # Remove older versions of this document first; the single ingestion job
        # below syncs both the deletions and the new objects, so the KB never
        # serves the old and new version of the same document at once.
        for old_content_hash in (superseded or []):
            self._delete_document(old_content_hash)

        total = len(chunks)
        for i, chunk in enumerate(chunks):
            # Content-addressed: stable id from the embedded text. Keeps the
            # {content_hash}_ prefix so identical chunks stay per-document (citation-safe),
            # while deduping exact-duplicate chunks within a document across re-runs.
            chunk_id = hashlib.sha256(chunk.enriched_text.encode()).hexdigest()[:12]
            key = f"chunks/{PIPELINE_CONFIG_HASH}/{chunk.content_hash}_{chunk_id}.txt"
            self._s3.put_object(Bucket=S3_BUCKET, Key=key, Body=chunk.enriched_text.encode())
            meta = {"metadataAttributes": {
                "raw_text":            chunk.text,
                "headings":            chunk.headings,
                "parent_headings":     chunk.parent_headings,
                "summary":             chunk.summary,
                "filename":            chunk.filename,
                "content_hash":        chunk.content_hash,   # document id (group chunks by source)
                "chunk_id":            chunk_id,             # stable per-chunk id
                "pipeline_config_hash": PIPELINE_CONFIG_HASH,
                "chunk_index":         i,       # position in document (reading order)
                "total_chunks":        total,   # for "chunk i of N" completeness
                "page_numbers":        ",".join(str(p) for p in chunk.page_numbers),  # source-page citation
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

    def _delete_document(self, content_hash: str) -> None:
        """Delete all S3 objects (chunks + metadata sidecars) for one document
        version under the current config prefix."""
        prefix = f"chunks/{PIPELINE_CONFIG_HASH}/{content_hash}_"
        to_delete = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            to_delete.extend({"Key": obj["Key"]} for obj in page.get("Contents", []))
        for i in range(0, len(to_delete), 1000):  # delete_objects caps at 1000/call
            self._s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": to_delete[i:i + 1000]})
        if to_delete:
            print(f"[vectorstore] Superseded {content_hash[:10]}... — removed {len(to_delete)} objects.")

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
        # Bedrock KB has no Milvus-style expr filter (expr is ignored). Over-fetch,
        # then drop excluded headings client-side — mirrors the local backend's
        # RETRIEVAL_EXPR "headings != 'Contents'". 100 is Bedrock's max numberOfResults.
        resp = self._runtime.retrieve(
            knowledgeBaseId=BEDROCK_KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": min(k * 2, 100)},
            },
        )
        docs = []
        for r in resp["retrievalResults"]:
            meta = r["metadata"]
            if meta.get("headings", "") in RETRIEVAL_EXCLUDE_HEADINGS:
                continue
            # Bedrock KB is one shared index over all past configs; keep only the
            # current pipeline config (local Milvus gets this via a per-config .db).
            if meta.get("pipeline_config_hash") != PIPELINE_CONFIG_HASH:
                continue
            docs.append(Document(
                page_content=r["content"]["text"],
                metadata={
                    "raw_text":        meta.get("raw_text", ""),
                    "headings":        meta.get("headings", ""),
                    "parent_headings": meta.get("parent_headings", ""),
                    "summary":         meta.get("summary", ""),
                    "filename":        meta.get("filename", ""),
                    "content_hash":    meta.get("content_hash", ""),
                    "chunk_id":        meta.get("chunk_id", ""),
                    "chunk_index":     meta.get("chunk_index", -1),
                    "total_chunks":    meta.get("total_chunks", -1),
                    "page_numbers":    [int(p) for p in meta.get("page_numbers", "").split(",") if p],
                },
            ))
            if len(docs) == k:
                break
        return docs
