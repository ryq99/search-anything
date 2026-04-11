import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from docling.document_converter import DocumentConverter
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_core.documents import Document
import anthropic
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from rag.config import (
    DATA_DIR, VECTOR_STORE_DIR, PROCESSED_BOOKS_PATH,
    EMBED_MODEL_ID, VECTOR_STORE_NAME, MILVUS_URI,
    CHUNK_MAX_TOKENS, TOC_MODEL, SUMMARY_MODEL,
    SUMMARY_SEMAPHORE, SUMMARY_MAX_TOKENS,
    BOOKS_DIR, ANTHROPIC_API_KEY,
)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_processed_books() -> dict:
    if PROCESSED_BOOKS_PATH.exists():
        return json.loads(PROCESSED_BOOKS_PATH.read_text())
    return {"books": {}}


def save_processed_books(registry: dict) -> None:
    PROCESSED_BOOKS_PATH.write_text(json.dumps(registry, indent=2))


def is_already_ingested(binary_hash: str, registry: dict) -> bool:
    return binary_hash in registry.get("books", {})


# ---------------------------------------------------------------------------
# Artifact naming
# ---------------------------------------------------------------------------

def _stem(pdf_path: Path) -> str:
    return pdf_path.stem.replace(" ", "_").replace("-", "_").lower()


# ---------------------------------------------------------------------------
# Step 1: PDF → Markdown
# ---------------------------------------------------------------------------

def convert_pdf_to_markdown(pdf_path: Path) -> tuple[str, str]:
    """Convert PDF to markdown using docling. Returns (markdown_text, binary_hash)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    md_text = result.document.export_to_markdown()

    stem = _stem(pdf_path)
    out_file = DATA_DIR / f"{stem}_converted.md"
    out_file.write_text(md_text, encoding="utf-8")
    print(f"[ingestion] Saved markdown: {out_file}")

    # Extract binary_hash from docling result
    doc_dict = result.document.export_to_dict()
    binary_hash = doc_dict.get("origin", {}).get("binary_hash", "")
    if not binary_hash:
        # Fallback: use file content hash
        import hashlib
        binary_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    return md_text, binary_hash


# ---------------------------------------------------------------------------
# Step 2: TOC extraction
# ---------------------------------------------------------------------------

def extract_toc(markdown_text: str, stem: str) -> pd.DataFrame:
    """Extract hierarchical TOC from markdown using Claude. Saves *_toc.csv."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=TOC_MODEL,
        max_tokens=4096,
        system=(
            "Extract table-of-contents hierarchy from markdown. "
            "Each hierarchy level must include BOTH the section number "
            "and the section title. "
            "Format each path as: "
            "<number> <title> => <number> <title> => ... "
            "Use any depth. Output only the paths. "
            "Return an empty string if no table of contents is present."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the table of contents from the markdown below.\n"
                    "Include the section number and section title at every level.\n\n"
                    "Example:\n"
                    "1 Introduction => 1.2 Random Variables => 1.2.2 Distributions\n\n"
                    f"{markdown_text}"
                ),
            },
        ],
    )

    raw = response.content[0].text or ""
    toc_value = [
        row.strip().split(" => ")
        for row in raw.split("\n")
        if row.strip()
    ]

    if not toc_value:
        return pd.DataFrame()

    # Normalize rows to uniform width (pad shorter rows with "")
    max_depth = max(len(row) for row in toc_value)
    toc_value = [row + [""] * (max_depth - len(row)) for row in toc_value]

    toc = pd.DataFrame(
        toc_value,
        columns=[f"Level {i + 1}" for i in range(max_depth)],
    ).fillna("")

    out_file = DATA_DIR / f"{stem}_toc.csv"
    toc.to_csv(out_file, index=False)
    print(f"[ingestion] Saved TOC: {out_file}")
    return toc


# ---------------------------------------------------------------------------
# Step 3: Chunking + metadata enrichment
# ---------------------------------------------------------------------------

def chunk_and_enrich(
    md_path: Path,
    toc_df: pd.DataFrame,
    binary_hash: str,
    filename: str,
) -> tuple[list[Document], dict[str, str]]:
    """Load markdown with DoclingLoader + HybridChunker, enrich with TOC metadata."""
    loader = DoclingLoader(
        file_path=[str(md_path)],
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(
            tokenizer=EMBED_MODEL_ID,
            max_tokens=CHUNK_MAX_TOKENS,
        ),
    )
    docs = loader.load()

    splits = []
    parent_headings_text: dict[str, str] = {}

    for doc in docs:
        dl_meta = doc.metadata.get("dl_meta", {})
        headings_list = dl_meta.get("headings", [])

        # Guard: skip chunks with no heading info
        if not headings_list:
            splits.append(Document(
                page_content=doc.page_content,
                metadata={
                    "binary_hash": binary_hash,
                    "filename": filename,
                    "headings": "",
                    "parent_headings": "",
                    "child_headings": "",
                },
            ))
            continue

        heading = headings_list[0]
        nth_level = len(heading.split(" ")[0].split("."))

        parent_headings_val = []
        child_headings_val = []

        if not toc_df.empty and nth_level <= toc_df.shape[1]:
            level_col = f"Level {nth_level}"
            if level_col in toc_df.columns:
                matched = toc_df.loc[
                    toc_df[level_col].str.contains(heading, regex=False)
                ]
                if not matched.empty:
                    parent_cols = [f"Level {i + 1}" for i in range(nth_level - 1)]
                    if parent_cols:
                        parent_headings_val = sorted(
                            matched[parent_cols]
                            .drop_duplicates()
                            .values.reshape(-1)
                            .tolist()
                        )
                    child_cols = [
                        f"Level {i + 1}"
                        for i in range(nth_level, toc_df.shape[1])
                        if f"Level {i + 1}" in toc_df.columns
                    ]
                    if child_cols:
                        child_headings_val = sorted(
                            matched[child_cols]
                            .drop_duplicates()
                            .values.reshape(-1)
                            .tolist()
                        )

        parent_headings_str = " => ".join(h for h in parent_headings_val if h)
        child_headings_str = " => ".join(h for h in child_headings_val if h)
        headings_str = " => ".join(headings_list)

        splits.append(Document(
            page_content=doc.page_content,
            metadata={
                "binary_hash": binary_hash,
                "filename": filename,
                "headings": headings_str,
                "parent_headings": parent_headings_str,
                "child_headings": child_headings_str,
            },
        ))

        # Accumulate text for parent heading summarization
        if parent_headings_str:
            if parent_headings_str not in parent_headings_text:
                parent_headings_text[parent_headings_str] = doc.page_content
            else:
                parent_headings_text[parent_headings_str] += "\n" + doc.page_content

    print(f"[ingestion] Created {len(splits)} chunks, {len(parent_headings_text)} parent heading groups")
    return splits, parent_headings_text


# ---------------------------------------------------------------------------
# Step 4: Async parent heading summarization
# ---------------------------------------------------------------------------

async def _summarize_one(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    parent_headings: str,
    text: str,
) -> tuple[str, str]:
    async with semaphore:
        response = await client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=SUMMARY_MAX_TOKENS,
            system=(
                "Summarize the book contents with condensed form in 3-4 sentences."
                "Cover the main topics, definition, proof, methods, and applications if any."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize the following text:\n\n{text}",
                },
            ],
        )
        return parent_headings, response.content[0].text or ""


async def summarize_all_parent_headings(
    parent_headings_text: dict[str, str],
) -> dict[str, str]:
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    semaphore = asyncio.Semaphore(SUMMARY_SEMAPHORE)
    tasks = [
        _summarize_one(client, semaphore, ph, text)
        for ph, text in parent_headings_text.items()
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)


def save_summaries_csv(summaries: dict[str, str], stem: str) -> Path:
    out_file = DATA_DIR / f"{stem}_parent_headings_summary.csv"
    pd.DataFrame(
        summaries.items(), columns=["Parent Headings", "Summary"]
    ).to_csv(out_file, index=False)
    print(f"[ingestion] Saved summaries: {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Step 5: Embed + store
# ---------------------------------------------------------------------------

def embed_and_store(splits: list[Document], drop_old: bool = False) -> Milvus:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    embedding = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"trust_remote_code": True},
    )

    if not Path(MILVUS_URI).exists() or drop_old:
        print("[ingestion] Creating new Milvus collection...")
        return Milvus.from_documents(
            documents=splits,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            connection_args={"uri": MILVUS_URI},
            index_params={"index_type": "FLAT"},
            drop_old=drop_old,
        )
    else:
        print("[ingestion] Adding documents to existing Milvus collection...")
        vs = Milvus(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            connection_args={"uri": MILVUS_URI},
            index_params={"index_type": "FLAT"},
        )
        vs.add_documents(splits)
        return vs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: Path) -> dict:
    """Full ingestion pipeline for a single PDF. Idempotent."""
    registry = load_processed_books()
    stem = _stem(pdf_path)

    print(f"[ingestion] Converting {pdf_path.name}...")
    md_text, binary_hash = convert_pdf_to_markdown(pdf_path)

    if is_already_ingested(binary_hash, registry):
        print(f"[ingestion] Already ingested: {pdf_path.name} (hash={binary_hash[:8]}...)")
        return registry["books"][binary_hash]

    md_path = DATA_DIR / f"{stem}_converted.md"

    print(f"[ingestion] Extracting TOC...")
    toc_df = extract_toc(md_text, stem)

    print(f"[ingestion] Chunking and enriching...")
    splits, parent_headings_text = chunk_and_enrich(
        md_path, toc_df, binary_hash, pdf_path.name
    )

    print(f"[ingestion] Summarizing {len(parent_headings_text)} parent heading groups...")
    summaries = asyncio.run(summarize_all_parent_headings(parent_headings_text))
    summary_csv = save_summaries_csv(summaries, stem)

    print(f"[ingestion] Embedding and storing {len(splits)} chunks...")
    embed_and_store(splits, drop_old=False)

    entry = {
        "filename": pdf_path.name,
        "pdf_path": str(pdf_path.absolute()),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "toc_csv": str(DATA_DIR / f"{stem}_toc.csv"),
        "summary_csv": str(summary_csv),
        "chunk_count": len(splits),
    }
    registry["books"][binary_hash] = entry
    save_processed_books(registry)

    print(f"[ingestion] Done: {pdf_path.name} ({len(splits)} chunks)")
    return entry


def ingest_books(books_dir: Path | None = None) -> list[dict]:
    """Scan books_dir for all *.pdf files and ingest any that are new."""
    books_dir = books_dir or BOOKS_DIR
    books_dir.mkdir(parents=True, exist_ok=True)
    registry = load_processed_books()
    results = []
    for pdf_path in sorted(books_dir.glob("*.pdf")):
        # Quick pre-check by filename to avoid full PDF conversion when possible
        already = any(
            e["filename"] == pdf_path.name
            for e in registry["books"].values()
        )
        if already:
            print(f"[ingestion] Skipping (already ingested): {pdf_path.name}")
            continue
        results.append(ingest_pdf(pdf_path))
    return results
