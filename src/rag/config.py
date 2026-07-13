import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Backend ────────────────────────────────────────────────────────────────
CLOUD_BACKEND = os.getenv("CLOUD_BACKEND", "local")  # "local" | "aws"

# ── Paths (local mode) ─────────────────────────────────────────────────────
# config.py lives at src/rag/config.py, so three .parent hops reach the repo root
# (config.py -> rag -> src -> repo root) where books/, data/, vector_store/ live.
PROJECT_ROOT         = Path(__file__).parent.parent.parent
BOOKS_DIR            = PROJECT_ROOT / "books"
DATA_DIR             = PROJECT_ROOT / "data"
VECTOR_STORE_DIR     = PROJECT_ROOT / "vector_store"
PROCESSED_BOOKS_PATH = PROJECT_ROOT / "processed_books.json"

# ── Ingestion ────────────────────────────────────────────────────────────────

# --- Parsing ---
LOCAL_PARSER = os.getenv("LOCAL_PARSER", "docling")  # "liteparse" | "docling"
PARSER_ENABLE_OCR = os.getenv("PARSER_ENABLE_OCR", "true").lower() == "true"
# Below this many chars, a parse is treated as a likely scanned/empty doc.
PARSER_MIN_CONTENT_LENGTH = 500

# --- Chunking ---
LOCAL_CHUNKER    = os.getenv("LOCAL_CHUNKER", "docling")  # "liteparse" | "docling"
# Tokenizer that drives token counting + the model's max sequence length. Keep
# this equal to the embedding model so chunk sizes match what actually gets embedded.
CHUNK_TOKENIZER  = os.getenv("CHUNK_TOKENIZER", "Snowflake/snowflake-arctic-embed-l-v2.0")
# Hard ceiling on tokens per chunk (the token-aware split pass). Oversized
# doc-items get split until each piece fits under this limit.
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "1024"))
# Merge undersized adjacent chunks that share the same heading path (3rd pass).
# True = fewer, denser chunks; False = preserve original doc-item boundaries.
CHUNK_MERGE_PEERS = os.getenv("CHUNK_MERGE_PEERS", "true").lower() == "true"
# Collapse consecutive list items into a single chunk (hierarchical 1st pass).
# True = a bullet list stays together; False = each item is its own chunk.
CHUNK_MERGE_LIST_ITEMS = os.getenv("CHUNK_MERGE_LIST_ITEMS", "true").lower() == "true"

# ── Pipeline config hash ──────────────────────────────────────────────────────
# Uniquely identifies a full ingestion pipeline configuration — every method
# decision that affects the stored artifact (parse → chunk → summarize → embed).
# Changing any of these parameters produces a new hash, a new registry namespace,
# and a new vector store partition, so stale artifacts are never silently reused.
_pipeline_config = {
    "parser":                LOCAL_PARSER,
    "parser_ocr":            PARSER_ENABLE_OCR,
    "chunker":               LOCAL_CHUNKER,
    "chunk_tokenizer":       CHUNK_TOKENIZER,
    "chunk_max_tokens":      CHUNK_MAX_TOKENS,
    "chunk_merge_peers":     CHUNK_MERGE_PEERS,
    "chunk_merge_list_items": CHUNK_MERGE_LIST_ITEMS,
    "embed_model":           "Snowflake/snowflake-arctic-embed-l-v2.0"
                             if CLOUD_BACKEND == "local"
                             else os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"),
    "summary_model":         os.getenv("LOCAL_SUMMARY_MODEL", "gemma4:e4b")
                             if CLOUD_BACKEND == "local"
                             else os.getenv("CLOUD_SUMMARY_MODEL", "claude-haiku-4-5-20251001"),
}
PIPELINE_CONFIG_HASH = hashlib.sha256(
    json.dumps(_pipeline_config, sort_keys=True).encode()
).hexdigest()[:12]

# ── Indexing ─────────────────────────────────────────────────────────────────

# --- Embedding ---
EMBED_MODEL_ID    = "Snowflake/snowflake-arctic-embed-l-v2.0"
VECTOR_STORE_NAME = f"rag_{PIPELINE_CONFIG_HASH}"
MILVUS_URI        = str(VECTOR_STORE_DIR / f"{PIPELINE_CONFIG_HASH}.db")

# --- Summarization ---
# Chunk summarization runs at indexing time to enrich each chunk before storage.
LOCAL_LLM_BASE_URL  = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
LOCAL_SUMMARY_MODEL = os.getenv("LOCAL_SUMMARY_MODEL", "gemma4:e4b")
CLOUD_SUMMARY_MODEL = os.getenv("CLOUD_SUMMARY_MODEL", "us.anthropic.claude-haiku-4-5-20251001")
SUMMARY_SEMAPHORE   = 5
SUMMARY_MAX_TOKENS  = 1000

# --- Vector Index ---
# Index algorithm: FLAT = exact brute-force; HNSW = fast approximate (recommended for scale);
# IVF_FLAT / IVF_SQ8 / IVF_PQ = cluster-based approximate.
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "FLAT")  # FLAT | HNSW | IVF_FLAT | IVF_SQ8 | IVF_PQ

# Distance metric. Arctic-embed is trained with cosine similarity — use COSINE.
# L2 = Euclidean distance; IP = inner product (dot product).
MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "COSINE")  # COSINE | L2 | IP

# HNSW params (only used when MILVUS_INDEX_TYPE=HNSW)
# M: max bi-directional links per node. Higher = better recall, more memory. Range: 4–64.
MILVUS_HNSW_M = int(os.getenv("MILVUS_HNSW_M", "16"))
# efConstruction: build-time search scope. Higher = better index quality, slower build. Range: 8–512.
MILVUS_HNSW_EF_CONSTRUCTION = int(os.getenv("MILVUS_HNSW_EF_CONSTRUCTION", "200"))

# IVF params (only used when MILVUS_INDEX_TYPE=IVF_*)
# nlist: number of Voronoi clusters. Typical rule of thumb: sqrt(n_vectors). Range: 1–65536.
MILVUS_IVF_NLIST = int(os.getenv("MILVUS_IVF_NLIST", "128"))

# ── Inference ────────────────────────────────────────────────────────────────

# --- Retrieval ---
RETRIEVAL_K    = int(os.getenv("RETRIEVAL_K", "10"))
RETRIEVAL_EXPR = os.getenv("RETRIEVAL_EXPR", "headings != 'Contents'")
# AWS KB retrieval has no native Milvus-style expr filter; this drives a
# client-side post-filter that mirrors RETRIEVAL_EXPR's "headings != 'Contents'".
RETRIEVAL_EXCLUDE_HEADINGS = {
    h.strip() for h in os.getenv("RETRIEVAL_EXCLUDE_HEADINGS", "Contents").split(",") if h.strip()
}

# --- Synthesis ---
# Answer synthesis runs at query time to generate the final response.
LOCAL_SYNTHESIS_MODEL  = os.getenv("LOCAL_SYNTHESIS_MODEL", "gemma4:e4b")
CLOUD_SYNTHESIS_MODEL  = os.getenv("CLOUD_SYNTHESIS_MODEL", "us.anthropic.claude-sonnet-4-6-20250514")
SYNTHESIS_MAX_TOKENS   = 8192

# ── Infrastructure ────────────────────────────────────────────────────────────

# --- AWS backend ---
AWS_REGION                = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET                 = os.getenv("S3_BUCKET", "")
DYNAMODB_TABLE            = os.getenv("DYNAMODB_TABLE", "")
BEDROCK_REGION            = os.getenv("BEDROCK_REGION", "us-west-2")
BEDROCK_KNOWLEDGE_BASE_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
BEDROCK_DATA_SOURCE_ID    = os.getenv("BEDROCK_DATA_SOURCE_ID", "")
BEDROCK_EMBED_MODEL_ID    = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")


# --- Secrets ---
def _load_secret(env_var: str, ssm_name: str) -> str:
    """Env var if set, else fetch from AWS SSM Parameter Store and export it so
    libraries that read the env directly (e.g. huggingface_hub) pick it up.
    Fails soft — missing boto3/creds/param yields "" and never breaks a run."""
    val = os.getenv(env_var, "")
    if val:
        return val
    try:
        import boto3
        val = boto3.client("ssm", region_name=AWS_REGION).get_parameter(
            Name=ssm_name, WithDecryption=True,
        )["Parameter"]["Value"]
        os.environ[env_var] = val
        return val
    except Exception:
        return ""


HF_TOKEN          = _load_secret("HF_TOKEN", "hf_hub_access_token")
ANTHROPIC_API_KEY = _load_secret("ANTHROPIC_API_KEY", "anthropic_api_key")

# KB ingestion job sync polling — store() blocks until the job reaches a terminal state.
KB_SYNC_POLL_INTERVAL     = int(os.getenv("KB_SYNC_POLL_INTERVAL", "5"))    # seconds between status checks
KB_SYNC_TIMEOUT           = int(os.getenv("KB_SYNC_TIMEOUT", "600"))        # max wait before giving up

os.environ["TOKENIZERS_PARALLELISM"] = "false"
