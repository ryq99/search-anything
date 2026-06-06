# Search Anything

A local-first RAG (Retrieval-Augmented Generation) system for indexing and querying documents — optimized for ML books and technical content.

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, Markdown, and plain text
- **Pluggable parsers** — `docling` (ML layout model, default) or `liteparse` (fast native Rust + Tesseract OCR), switchable via `LOCAL_PARSER`
- **Structure-native chunking** — DoclingChunker operates on the live DoclingDocument tree (no markdown round-trip); heading ancestry is derived directly from docling's semantic structure
- **Heading-contextualized embeddings** — `chunker.contextualize()` prefixes each chunk with its heading path before embedding, materially improving retrieval quality for section-level queries
- **Section summaries** — async per-heading summaries stored alongside chunks for richer retrieval context
- **Idempotent pipeline** — SHA-256 content hash prevents double-ingestion
- **File watcher** — `watch` command auto-ingests files dropped into `books/`
- **Dual backend** — `local` (Milvus Lite + Anthropic API) or `aws` (Bedrock + DynamoDB), switched via `CLOUD_BACKEND`

## Setup

```bash
python -m venv .venv && source .venv/bin/activate

pip install -e .
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

## Workflow

Indexing converts source files into searchable vector chunks. Each file goes through four stages:

```mermaid
%%{init: {'look': 'handDrawn'}}%%
flowchart TD
    A("Source file\nPDF / DOCX / PPTX / MD / TXT") --> B

    subgraph B["1. Parse"]
        B1("docling (default)\nML layout model → DoclingDocument\npreserves headings, tables, reading order")
        B2("liteparse (LOCAL_PARSER=liteparse)\nfast Rust core + Tesseract OCR\noutputs markdown")
        B3("plain text / MD\npassthrough")
    end

    B --> C("ParseResult\nmarkdown + SHA-256 content hash\n+ DoclingDocument (docling path)")

    C --> D

    subgraph D["2. Chunk + Enrich"]
        D1("HierarchicalChunker\nsplit on headings & lists")
        D2("Token-aware split\nbreak chunks > CHUNK_MAX_TOKENS")
        D3("merge_peers\nmerge undersized same-heading neighbours")
        D1 --> D2 --> D3
    end

    D --> E("Chunk objects\ntext + enriched_text (heading-prefixed) + metadata\nparent_headings, headings derived from document tree")

    E --> F("3. Section Summarization\nClaude summarizes each parent-heading group\nconcurrently (semaphore=5)\n→ summaries CSV in data/")

    F --> G("4. Embed + Store\nSnowflake Arctic Embed L v2.0\nembeds enriched_text of each chunk")

    G --> H1("Milvus Lite\n(local)")
    G --> H2("Bedrock Knowledge Base\n(AWS)")

    F --> I1("processed_books.json\n(local registry)")
    F --> I2("DynamoDB\n(AWS registry)")
```

### Ingest commands

```bash
# Ingest all new files in books/
python main.py ingest
# or, if installed as a package:
search-anything ingest

# Ingest specific files
python main.py ingest --paths books/deeplearning.pdf books/notes.md

# Watch books/ and auto-ingest on file creation (performs catch-up on startup)
python main.py watch
```

Files already present in the registry (matched by content hash) are skipped automatically.

## Configuration

All tunables are in [src/rag/config.py](src/rag/config.py) and overridable via `.env`. See [.env.example](.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `CLOUD_BACKEND` | `local` | `local` or `aws` |
| `LOCAL_PARSER` | `docling` | `docling` (ML layout) or `liteparse` (fast Rust + OCR) |
| `PARSER_ENABLE_OCR` | `true` | OCR on scanned/mixed pages (docling path) |
| `LOCAL_CHUNKER` | `docling` | Chunker strategy (currently `docling` only) |
| `CHUNK_MAX_TOKENS` | `1024` | Hard token ceiling per chunk |
| `CHUNK_MERGE_PEERS` | `true` | Merge undersized same-heading neighbours |
| `CHUNK_MERGE_LIST_ITEMS` | `true` | Collapse consecutive list items into one chunk |
| `EMBED_MODEL_ID` | `Snowflake/snowflake-arctic-embed-l-v2.0` | HuggingFace embedding model |
| `RETRIEVAL_K` | `10` | Number of chunks to retrieve |
| `SYNTHESIS_MODEL` | `claude-haiku-4-5-20251001` | Model for answer synthesis |
