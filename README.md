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

## Pipelines

### Ingestion

```mermaid
%%{init: {'look': 'handDrawn'}}%%
flowchart LR
    A("Source file
    PDF / DOCX / PPTX / MD / TXT")

    subgraph PARSE["1 · Parse"]
        direction TB
        P1("docling
        ML layout → DoclingDocument")
        P2("liteparse
        Rust + OCR → markdown")
        P3("plaintext
        passthrough")
    end

    B("ParseResult
    markdown · content_hash · doc_dir")

    subgraph CHUNK["2 · Chunk"]
        direction TB
        C1("DoclingChunker
        HybridChunker on document tree
        heading ancestry + enriched_text")
        C2("LiteParseChunker
        paragraph split → sentence fallback
        token-bounded greedy merge")
    end

    D("list[Chunk]
    text · enriched_text · headings")

    subgraph SUM["3 · Summarize"]
        direction TB
        S1("Gemma4 via Ollama
        local")
        S2("Claude Haiku
        cloud")
    end

    E("Enriched Chunks
    + summary populated per chunk")

    A --> PARSE --> B --> CHUNK --> D --> SUM --> E
```

### Indexing

```mermaid
%%{init: {'look': 'handDrawn'}}%%
flowchart LR
    A("Enriched Chunks
    from ingestion")

    B{"Already
    indexed?"}

    SKIP(("skip"))

    C("4 · Embed
    Snowflake Arctic Embed L v2.0
    1024-dim · COSINE metric")

    subgraph STORE["5 · Store"]
        direction TB
        S1("Milvus Lite
        local .db file")
        S2("Bedrock KB
        AWS")
    end

    subgraph REG["5 · Register"]
        direction TB
        R1("processed_books.json
        local registry")
        R2("DynamoDB
        AWS registry")
    end

    A --> B
    B -- "yes" --> SKIP
    B -- "no" --> C --> STORE
    C --> REG
```

### Inference

```mermaid
%%{init: {'look': 'handDrawn'}}%%
flowchart LR
    Q("Question")

    R("6 · Retrieve
    similarity search
    top-K=10 · filter headings ≠ Contents")

    subgraph VS["Vector Store"]
        direction TB
        V1("Milvus Lite
        local")
        V2("Bedrock KB
        AWS")
    end

    F("Retrieved Chunks
    rank · headings · summary · text")

    subgraph SYN["7 · Synthesize"]
        direction TB
        Y1("Gemma4 via Ollama
        local")
        Y2("Claude Sonnet
        cloud")
    end

    ANS("Answer")

    Q --> R
    VS --> R
    R --> F --> SYN --> ANS
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
