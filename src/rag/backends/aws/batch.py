"""Bedrock Batch Inference helpers for summarization (~50% cheaper than real-time).

Pure formatters (record/JSONL/job-request/output parsing) plus `run_summary_batch`,
the boto3 runner that ties them together. Batch uses the native Anthropic Messages
request schema (not `converse`), so structured output is expressed via `tools`/`tool_choice`.
"""
import json
import time
import uuid


def build_summary_record(record_id: str, system: str, user: str, schema: dict, max_tokens: int) -> dict:
    """One Batch Inference JSONL record: a structured (tool-use) summary request for
    one chunk. `record_id` correlates the result back to the chunk (Bedrock requires
    it to be at least 11 alphanumeric chars — the chunk's content-addressed id fits)."""
    return {
        "recordId": record_id,
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
            "tools": [{"name": "emit", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "emit"},
        },
    }


def build_batch_jsonl(records: list[dict]) -> str:
    """Serialize records to the JSONL body uploaded to S3 as the batch job input."""
    return "\n".join(json.dumps(r) for r in records)


def build_job_request(job_name: str, model_id: str, role_arn: str, input_uri: str, output_uri: str) -> dict:
    """Kwargs for bedrock `create_model_invocation_job`. `input_uri` points at the
    input JSONL in S3; `output_uri` is the output prefix Bedrock writes results to."""
    return {
        "jobName": job_name,
        "modelId": model_id,
        "roleArn": role_arn,
        "inputDataConfig": {"s3InputDataConfig": {"s3Uri": input_uri}},
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": output_uri}},
    }


def parse_summary_output(jsonl: str) -> dict[str, str]:
    """Map recordId -> summary from a Bedrock Batch Inference output JSONL.

    Each line is {"recordId", "modelOutput": {anthropic response}} on success; the
    summary is the `emit` tool_use block's input. Records that errored or lack a
    tool_use block are omitted so the caller can apply its truncated-text fallback.
    """
    summaries: dict[str, str] = {}
    for line in jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rid = rec.get("recordId")
        model_output = rec.get("modelOutput")
        if not rid or not model_output:
            continue
        for block in model_output.get("content", []):
            if block.get("type") == "tool_use" and "summary" in block.get("input", {}):
                summaries[rid] = block["input"]["summary"]
                break
    return summaries


_TERMINAL_OK = {"Completed", "PartiallyCompleted"}
_TERMINAL_BAD = {"Failed", "Expired", "Stopped"}


def run_summary_batch(chunk_texts: list[str], system: str, schema: dict, max_tokens: int,
                      poll_interval: int = 30, timeout: int = 7200) -> dict[int, str]:
    """Run one Bedrock Batch Inference job to summarize `chunk_texts`.

    Uploads a JSONL of tool-use records to S3, starts the job, blocks until it reaches
    a terminal state, then parses the output. Returns {index -> summary} for records
    that succeeded (caller applies a fallback for any missing index). Record ids are
    positional (`rec{i}`) so results map straight back to the input order.
    """
    import boto3
    from rag.config import (
        AWS_REGION, BEDROCK_REGION, S3_BUCKET, BEDROCK_BATCH_ROLE_ARN,
        BEDROCK_BATCH_S3_PREFIX, CLOUD_SUMMARY_MODEL,
    )

    s3 = boto3.client("s3", region_name=AWS_REGION)
    bedrock = boto3.client("bedrock", region_name=BEDROCK_REGION)

    job_name = f"sa-sum-{uuid.uuid4().hex[:12]}"
    base = f"{BEDROCK_BATCH_S3_PREFIX.rstrip('/')}/{job_name}"
    input_key = f"{base}/input.jsonl"
    output_prefix = f"{base}/output/"

    records = [
        build_summary_record(f"rec{i:09d}", system, text, schema, max_tokens)
        for i, text in enumerate(chunk_texts)
    ]
    s3.put_object(Bucket=S3_BUCKET, Key=input_key, Body=build_batch_jsonl(records).encode())

    req = build_job_request(
        job_name, CLOUD_SUMMARY_MODEL, BEDROCK_BATCH_ROLE_ARN,
        f"s3://{S3_BUCKET}/{input_key}", f"s3://{S3_BUCKET}/{output_prefix}",
    )
    job_arn = bedrock.create_model_invocation_job(**req)["jobArn"]
    print(f"[batch] summarization job {job_name} started — waiting (poll {poll_interval}s)...")

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = bedrock.get_model_invocation_job(jobIdentifier=job_arn)["status"]
        if status in _TERMINAL_OK:
            break
        if status in _TERMINAL_BAD:
            raise RuntimeError(f"Bedrock batch job {job_name} ended {status}")
        print(f"[batch] status: {status}")
        time.sleep(poll_interval)
    else:
        raise TimeoutError(f"Bedrock batch job {job_name} did not finish within {timeout}s")

    # Bedrock writes results (+ a manifest) under the output prefix; the records file
    # ends with .jsonl.out. Concatenate any such files and parse.
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=output_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl.out"):
                out.append(s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read().decode())
    by_id = parse_summary_output("\n".join(out))
    print(f"[batch] job {job_name} complete — {len(by_id)}/{len(chunk_texts)} summaries.")
    return {int(rid[3:]): summary for rid, summary in by_id.items()}
