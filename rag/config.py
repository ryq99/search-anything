from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- Paths ---
PROJECT_ROOT         = Path(__file__).parent.parent
BOOKS_DIR            = PROJECT_ROOT / "books"
DATA_DIR             = PROJECT_ROOT / "data"
VECTOR_STORE_DIR     = PROJECT_ROOT / "vector_store"
PROCESSED_BOOKS_PATH = PROJECT_ROOT / "processed_books.json"

# --- Embedding ---
EMBED_MODEL_ID    = "Snowflake/snowflake-arctic-embed-l-v2.0"
VECTOR_STORE_NAME = "book_a_snowflake_arctic_embed"   # matches existing DB
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

# --- Secrets ---
HF_TOKEN          = os.environ["HF_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
