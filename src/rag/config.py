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
# Which local parser handles pdf/docx/pptx: "docling" (ML layout model) or
# "liteparse" (fast native spatial parser). Lets you A/B parse quality.
LOCAL_PARSER = os.getenv("LOCAL_PARSER", "docling")  # "docling" | "liteparse"
# OCR recovers text from scanned pages / embedded images. force_full_page=False
# means clean text-layer pages stay fast; only failed pages invoke OCR.
PARSER_ENABLE_OCR = os.getenv("PARSER_ENABLE_OCR", "true").lower() == "true"
# Below this many chars, a parse is treated as a likely scanned/empty doc.
PARSER_MIN_CONTENT_LENGTH = 500
# VLM image description (figures/diagrams). Slow + needs a VLM endpoint; later step.
PARSER_ENABLE_IMAGE_DESCRIPTION = False

# --- Chunking ---
LOCAL_CHUNKER    = os.getenv("LOCAL_CHUNKER", "docling")  # "docling" | future strategies
CHUNK_MAX_TOKENS = 1024

# --- Models ---
LLM_MODEL            = "claude-haiku-4-5-20251001"  # default for chunk/section summarization
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
