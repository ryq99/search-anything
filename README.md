# Search Anything

RAG for personal knowledge base.

## Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, Markdown, plain text
- **Pluggable parsers** — `docling` (ML layout model) or `liteparse` (Rust + Tesseract OCR)
- **Structure-native chunking** — DoclingChunker operates on the live DoclingDocument tree; LiteParseChunker uses paragraph-boundary splitting with sentence-level fallback
- **Heading-contextualized embeddings** — heading path prepended to each chunk before embedding
- **Per-chunk summaries** — Gemma4 (local) or Claude Haiku (cloud) summarizes each chunk at index time; summaries stored in Milvus and returned at retrieval
- **Idempotent pipeline** — SHA-256 content hash prevents double-ingestion
- **File watcher** — `watch` auto-ingests files dropped into `books/`
- **Dual backend** — `local` (Milvus Lite + Ollama) or `aws` (Bedrock + DynamoDB)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .           # local backend
pip install -e ".[aws]"    # + AWS backend
cp .env.example .env
```

Pull the local LLM:
```bash
ollama pull gemma4:e4b
```

## Indexing Pipeline

```mermaid
%%{init: {'look': 'handDrawn'}}%%
flowchart LR
    A("Source file
    PDF / DOCX / PPTX / MD / TXT")

    subgraph B["1. Parse"]
        direction TB
        B1("docling
        ML layout → DoclingDocument")
        B2("liteparse
        Rust + OCR → markdown")
        B3("plaintext
        passthrough")
    end

    C("ParseResult
    SHA-256 content hash")

    subgraph D["2. Chunk"]
        direction TB
        D1("DoclingChunker
        HybridChunker on document tree
        heading ancestry + enriched_text")
        D2("LiteParseChunker
        paragraph split → sentence fallback
        token-bounded greedy merge")
    end

    E("Chunk objects
    text · enriched_text · headings · summary")

    F("3. Summarize
    Gemma4 via Ollama — local
    Claude Haiku via Anthropic — cloud
    async · per chunk · stored in-place")

    G("4. Embed + Store
    Snowflake Arctic Embed L v2.0")

    H1("Milvus Lite
    local")
    H2("Bedrock KB
    AWS")
    I1("processed_books.json
    local registry")
    I2("DynamoDB
    AWS registry")

    A --> B --> C --> D --> E --> F --> G
    G --> H1 & H2
    G --> I1 & I2
```

## Commands

```bash
python main.py ingest                                    # ingest all files in books/
python main.py ingest --paths books/a.pdf books/b.pdf   # specific files
python main.py ask "What is gradient descent?"           # query
python main.py watch                                     # watch books/ and auto-ingest
```

Files already in the registry (matched by content hash + parser) are skipped.

## Configuration

All tunables are in [src/rag/config.py](src/rag/config.py), overridable via `.env`. See [.env.example](.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `CLOUD_BACKEND` | `local` | `local` or `aws` |
| `LOCAL_PARSER` | `liteparse` | `liteparse` or `docling` |
| `LOCAL_CHUNKER` | `liteparse` | `liteparse` or `docling` |
| `CHUNK_MAX_TOKENS` | `1024` | Hard token ceiling per chunk |
| `LOCAL_SUMMARY_MODEL` | `gemma4:e4b` | Ollama model for summarization (local) |
| `CLOUD_SUMMARY_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model for summarization (cloud) |
| `LOCAL_SYNTHESIS_MODEL` | `gemma4:e4b` | Ollama model for answer synthesis (local) |
| `CLOUD_SYNTHESIS_MODEL` | `claude-sonnet-4-6` | Anthropic model for answer synthesis (cloud) |
| `LOCAL_LLM_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `RETRIEVAL_K` | `10` | Chunks retrieved per query |
