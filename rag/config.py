from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- Backend switch ---
CLOUD_BACKEND = os.getenv("CLOUD_BACKEND", "local")  # "local" | "aws"

# --- Paths (local mode) ---
PROJECT_ROOT         = Path(__file__).parent.parent
BOOKS_DIR            = PROJECT_ROOT / "books"
DATA_DIR             = PROJECT_ROOT / "data"
VECTOR_STORE_DIR     = PROJECT_ROOT / "vector_store"
PROCESSED_BOOKS_PATH = PROJECT_ROOT / "processed_books.json"

# --- Embedding ---
EMBED_MODEL_ID    = "Snowflake/snowflake-arctic-embed-l-v2.0"
VECTOR_STORE_NAME = "book_a_snowflake_arctic_embed"
MILVUS_URI        = str(VECTOR_STORE_DIR / f"{VECTOR_STORE_NAME}.db")

# --- Chunking ---
CHUNK_MAX_TOKENS = 1024

# --- Models ---
TOC_MODEL            = "claude-haiku-4-5-20251001"
SUMMARY_MODEL        = "claude-haiku-4-5-20251001"
SYNTHESIS_MODEL      = os.getenv("SYNTHESIS_MODEL", "claude-haiku-4-5-20251001")
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
AWS_REGION         = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET          = os.getenv("S3_BUCKET", "")
DYNAMODB_TABLE     = os.getenv("DYNAMODB_TABLE", "")
AURORA_CONN_STRING = os.getenv("AURORA_CONN_STRING", "")
BEDROCK_REGION     = os.getenv("BEDROCK_REGION", "us-east-1")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
