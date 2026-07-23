"""Bedrock Batch Inference helpers for summarization (~50% cheaper than real-time).

Pure formatting only — no boto3/S3 here. The job runner (createModelInvocationJob,
poll, read output) builds on these. Batch uses the native Anthropic Messages request
schema (not `converse`), so structured output is expressed via `tools`/`tool_choice`.
"""
import json


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
