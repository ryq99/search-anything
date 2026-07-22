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
