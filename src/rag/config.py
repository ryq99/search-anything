from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- Backend switch ---
CLOUD_BACKEND = os.getenv("CLOUD_BACKEND", "local")  # "local" | "aws"

# --- Paths (local mode) ---
# config.py lives at src/rag/config.py, so three .parent hops reach the repo root
# (config.py -> rag -> src -> repo root) where books/, data/, vector_store/ live.
PROJECT_ROOT         = Path(__file__).parent.parent.parent
BOOKS_DIR            = PROJECT_ROOT / "books"
DATA_DIR             = PROJECT_ROOT / "data"
VECTOR_STORE_DIR     = PROJECT_ROOT / "vector_store"
PROCESSED_BOOKS_PATH = PROJECT_ROOT / "processed_books.json"

# --- Embedding ---
EMBED_MODEL_ID    = "Snowflake/snowflake-arctic-embed-l-v2.0"
VECTOR_STORE_NAME = "book_a_snowflake_arctic_embed"
MILVUS_URI        = str(VECTOR_STORE_DIR / f"{VECTOR_STORE_NAME}.db")

# --- Parsing ---
LOCAL_PARSER = os.getenv("LOCAL_PARSER", "liteparse")  # "liteparse" | "docling"
PARSER_ENABLE_OCR = os.getenv("PARSER_ENABLE_OCR", "true").lower() == "true" # force_full_page=False to avoid unnecessary OCR on clean PDFs
# Below this many chars, a parse is treated as a likely scanned/empty doc.
PARSER_MIN_CONTENT_LENGTH = 500
# VLM image description (figures/diagrams). Slow + needs a VLM endpoint; later step.
PARSER_ENABLE_IMAGE_DESCRIPTION = False

# --- Chunking ---
LOCAL_CHUNKER    = os.getenv("LOCAL_CHUNKER", "liteparse")  # "liteparse" | "docling"
# Tokenizer that drives token counting + the model's max sequence length. Keep
# this equal to the embedding model so chunk sizes match what actually gets embedded.
CHUNK_TOKENIZER  = os.getenv("CHUNK_TOKENIZER", EMBED_MODEL_ID)
# Hard ceiling on tokens per chunk (the token-aware split pass). Oversized
# doc-items get split until each piece fits under this limit.
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "1024"))
# Merge undersized adjacent chunks that share the same heading path (3rd pass).
# True = fewer, denser chunks; False = preserve original doc-item boundaries.
CHUNK_MERGE_PEERS = os.getenv("CHUNK_MERGE_PEERS", "true").lower() == "true"
# Collapse consecutive list items into a single chunk (hierarchical 1st pass).
# True = a bullet list stays together; False = each item is its own chunk.
CHUNK_MERGE_LIST_ITEMS = os.getenv("CHUNK_MERGE_LIST_ITEMS", "true").lower() == "true"

# --- Models ---
# Ollama base URL (local mode only)
LOCAL_LLM_BASE_URL       = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")

# Chunk summarization (indexing-time)
LOCAL_SUMMARY_MODEL      = os.getenv("LOCAL_SUMMARY_MODEL",  "gemma4:e4b")
CLOUD_SUMMARY_MODEL      = os.getenv("CLOUD_SUMMARY_MODEL",  "claude-haiku-4-5-20251001")

# Answer synthesis (query-time)
LOCAL_SYNTHESIS_MODEL    = os.getenv("LOCAL_SYNTHESIS_MODEL", "gemma4:e4b")
CLOUD_SYNTHESIS_MODEL    = os.getenv("CLOUD_SYNTHESIS_MODEL", "claude-sonnet-4-6")

SUMMARY_SEMAPHORE    = 5
SUMMARY_MAX_TOKENS   = 1000
SYNTHESIS_MAX_TOKENS = 8192

# --- Retrieval ---
RETRIEVAL_K    = 10
RETRIEVAL_EXPR = "headings != 'Contents'"

# --- Local backend secrets (validated at use, not at import) ---
HF_TOKEN          = os.getenv("HF_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- AWS backend config ---
AWS_REGION         = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET          = os.getenv("S3_BUCKET", "")
DYNAMODB_TABLE     = os.getenv("DYNAMODB_TABLE", "")
AURORA_CONN_STRING = os.getenv("AURORA_CONN_STRING", "")
BEDROCK_REGION     = os.getenv("BEDROCK_REGION", "us-west-2")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
