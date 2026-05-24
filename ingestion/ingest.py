"""
Document Ingestion Pipeline for Health Insurance Knowledge Base.

Loads PDFs and CSVs from the data/ folder, chunks them intelligently,
enriches with metadata and contextual headers, and persists to ChromaDB.

Usage:
    python -m ingestion.ingest
    # or from project root:
    python ingestion/ingest.py
"""

import sys
import os

# Ensure project root is on sys.path so `config` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import glob
import shutil
import pandas as pd
from transformers import AutoTokenizer
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from langchain_community.vectorstores.utils import filter_complex_metadata

from config import (
    DOCUMENTS_DIR,
    CHROMA_PERSIST_DIR,
    DOC_TYPE_MAP,
    PLAN_TIER_MAP,
    MAX_TOKENS_PER_CHUNK,
    CHUNK_OVERLAP,
    CSV_CHUNK_SIZE,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    RECORD_CSV_MIN_ROWS,
    RECORD_CSV_MIN_COLS,
)

console = Console()

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _classify_document(filename: str) -> dict:
    """Derive doc_type and plan_tier from the filename."""
    meta = {"doc_type": "unknown", "plan_tier": "all"}
    for key, doc_type in DOC_TYPE_MAP.items():
        if key in filename:
            meta["doc_type"] = doc_type
            break
    for key, tier in PLAN_TIER_MAP.items():
        if key in filename:
            meta["plan_tier"] = tier
            break
    return meta


def _is_table_chunk(text: str) -> bool:
    """Return True if the chunk contains a markdown table."""
    return any(line.strip().startswith("|") for line in text.splitlines())


def _is_record_style_csv(df: pd.DataFrame) -> bool:
    """
    Return True if this CSV looks like a record-per-row file
    (e.g. a drug formulary) rather than a summary/batch table.

    Heuristic: more than RECORD_CSV_MIN_ROWS rows AND every row
    has at least RECORD_CSV_MIN_COLS non-null values. When both
    are true, each row is an independent record and should be
    chunked individually instead of batched.
    """
    if len(df) < RECORD_CSV_MIN_ROWS:
        return False
    avg_non_null = df.notna().sum(axis=1).mean()
    return avg_non_null >= RECORD_CSV_MIN_COLS


def _build_contextual_header(metadata: dict) -> str:
    """
    Create a descriptive header for retrieval context.
    Stored in metadata['display_header'] — prepended at retrieval
    time by the retriever, NOT during embedding.
    """
    source = metadata.get("source_file", "Unknown")
    doc_type = metadata.get("doc_type", "Unknown").replace("_", " ").title()
    tier = metadata.get("plan_tier", "all")

    header = ">>> DOCUMENT CONTEXT <<<\n"
    header += f"Source: {source}\n"
    header += f"Document Type: {doc_type}\n"
    if tier != "all":
        header += f"Plan Tier: {tier}\n"
    if "drug_tier" in metadata:
        header += f"Drug Tier: {metadata['drug_tier']}\n"
    if "drug_class" in metadata:
        header += f"Drug Class: {metadata['drug_class']}\n"
    if "row_range" in metadata:
        header += f"CSV Rows: {metadata['row_range']}\n"
    if "group_value" in metadata:
        header += f"Group: {metadata['group_value']}\n"
    header += "------------------------\n"
    return header


# ──────────────────────────────────────────────────────────────
# CSV chunking strategies
# ──────────────────────────────────────────────────────────────

def _chunk_record_csv(df: pd.DataFrame, filename: str, classification: dict) -> list[Document]:
    """
    One Document per row for record-style CSVs like a drug formulary.

    Each row becomes a natural-language sentence:
        drug_name: metformin | brand_name: Glucophage | tier: 1 | ...

    Key columns (tier, drug_class, prior_auth_required) are also
    stored as filterable metadata fields in ChromaDB so the retriever
    can pre-filter before doing semantic search.
    """
    docs = []
    for idx, row in df.iterrows():
        content = " | ".join(
            f"{k}: {v}" for k, v in row.items() if pd.notna(v)
        )

        metadata = {
            "source_file": filename,
            "doc_type": classification["doc_type"],
            "plan_tier": classification["plan_tier"],
            "chunk_type": "record",
            "row_index": int(idx),
        }

        for col, meta_key in [
            ("tier",                 "drug_tier"),
            ("drug_class",           "drug_class"),
            ("prior_auth_required",  "prior_auth_required"),
            ("step_therapy_required","step_therapy_required"),
            ("formulary_status",     "formulary_status"),
            ("specialty_pharmacy_only", "specialty_pharmacy_only"),
        ]:
            if col in row and pd.notna(row[col]):
                metadata[meta_key] = str(row[col])

        metadata["display_header"] = _build_contextual_header(metadata)
        docs.append(Document(page_content=content, metadata=metadata))

    return docs


def _chunk_batch_csv(df: pd.DataFrame, filename: str, classification: dict) -> list[Document]:
    """
    Batch chunking for summary/tabular CSVs that don't have one
    meaningful record per row.
    """
    docs = []

    group_col = next(
        (col for col in df.columns
         if any(k in col.lower() for k in ["plan", "tier", "category", "group"])),
        None
    )

    def _make_batch_doc(batch, offset, group_val=None):
        markdown_table = batch.to_markdown(index=False)
        row_details = "\n".join(
            " | ".join(f"{k}: {v}" for k, v in row.items() if pd.notna(v))
            for _, row in batch.iterrows()
        )
        content = (
            f"### TABLE DATA (Rows {offset+1}-{offset+len(batch)})\n\n"
            f"{markdown_table}\n\n### ROW DETAILS\n{row_details}"
        )
        metadata = {
            "source_file": filename,
            "row_range": f"{offset+1}-{offset+len(batch)}",
            "doc_type": classification["doc_type"],
            "plan_tier": classification["plan_tier"],
            "chunk_type": "table",
        }
        if group_val is not None:
            metadata["group_value"] = str(group_val)
        metadata["display_header"] = _build_contextual_header(metadata)
        return Document(page_content=content, metadata=metadata)

    if group_col:
        for val, group in df.groupby(group_col):
            for i in range(0, len(group), CSV_CHUNK_SIZE):
                batch = group.iloc[i: i + CSV_CHUNK_SIZE]
                docs.append(_make_batch_doc(batch, i, val))
    else:
        for i in range(0, len(df), CSV_CHUNK_SIZE):
            batch = df.iloc[i: i + CSV_CHUNK_SIZE]
            docs.append(_make_batch_doc(batch, i))

    return docs


# ──────────────────────────────────────────────────────────────
# Document Loading & Chunking
# ──────────────────────────────────────────────────────────────

def load_and_chunk_documents() -> list[Document]:
    """Load and chunk PDFs via Docling and CSVs via the right strategy."""
    pdf_files = sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf")))
    csv_files = sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.csv")))
    all_chunks = []

    console.print("\n⚙️  Configuring Docling tokenizer...")
    # Use a standard tokenizer for chunking since OpenAI models don't have HF tokenizers
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS_PER_CHUNK,
        overlap=CHUNK_OVERLAP,
    )

    if pdf_files:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Processing PDFs...", total=len(pdf_files))
            for file_path in pdf_files:
                filename = os.path.basename(file_path)
                progress.update(task, description=f"Chunking {filename}")
                try:
                    loader = DoclingLoader(
                        file_path=file_path,
                        export_type=ExportType.DOC_CHUNKS,
                        chunker=chunker,
                    )
                    chunks = loader.load()
                    classification = _classify_document(filename)
                    for chunk in chunks:
                        chunk.metadata["chunk_type"] = (
                            "table" if _is_table_chunk(chunk.page_content) else "prose"
                        )
                        chunk.metadata.update({"source_file": filename, **classification})
                        chunk.metadata["display_header"] = _build_contextual_header(chunk.metadata)

                        dl_meta = chunk.metadata.get("dl_meta", {})
                        if dl_meta:
                            page_no = None
                            if "doc_items" in dl_meta and dl_meta["doc_items"]:
                                prov = dl_meta["doc_items"][0].get("prov", [])
                                if prov:
                                    page_no = prov[0].get("page_no")
                            
                            if page_no is not None:
                                chunk.metadata["page"] = page_no - 1  # 0-indexed for internal consistency
                                chunk.metadata["page_no"] = page_no
                            
                            chunk.metadata["heading"] = (
                                dl_meta.get("headings", [""])[0]
                                if dl_meta.get("headings") else ""
                            )
                            bbox = dl_meta.get("bbox")
                            if bbox:
                                chunk.metadata["bbox"] = str(bbox)

                    all_chunks.extend(chunks)
                except Exception as e:
                    console.print(f"[bold red]⚠ Skipping {filename}:[/] {e}")
                progress.advance(task)

    if csv_files:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Processing CSVs...", total=len(csv_files))
            for file_path in csv_files:
                filename = os.path.basename(file_path)
                progress.update(task, description=f"Chunking {filename}")
                try:
                    classification = _classify_document(filename)
                    df = pd.read_csv(file_path)

                    if _is_record_style_csv(df):
                        console.print(f"  📋 {filename} → record-per-row mode ({len(df)} rows)")
                        chunks = _chunk_record_csv(df, filename, classification)
                    else:
                        console.print(f"  📋 {filename} → batch mode")
                        chunks = _chunk_batch_csv(df, filename, classification)

                    all_chunks.extend(chunks)
                except Exception as e:
                    console.print(f"[bold red]⚠ Skipping {filename}:[/] {e}")
                progress.advance(task)

    console.print(
        f"  ✅ Created [bold green]{len(all_chunks)}[/] chunks from "
        f"[bold]{len(pdf_files) + len(csv_files)}[/] files"
    )
    return all_chunks


# ──────────────────────────────────────────────────────────────
# Embedding & Storage
# ──────────────────────────────────────────────────────────────

def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and persist to ChromaDB."""
    chunks = filter_complex_metadata(chunks)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        console.print("  🗑️  Cleared previous ChromaDB")

    console.print(f"  📦 Embedding [bold]{len(chunks)}[/] chunks into ChromaDB...")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Embedding & indexing...", total=None)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        progress.update(task, description="Done!", completed=True)

    console.print(f"  ✅ ChromaDB persisted to [bold]{CHROMA_PERSIST_DIR}[/]")
    return vectorstore


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold magenta]═══════════════════════════════════════════════════[/]")
    console.print("[bold magenta]   Health Insurance Knowledge Base — Ingestion[/]")
    console.print("[bold magenta]═══════════════════════════════════════════════════[/]\n")

    start_time = time.time()
    chunks = load_and_chunk_documents()
    build_vector_store(chunks)
    elapsed = time.time() - start_time

    summary_table = Table(title="Knowledge Base Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Total chunks", str(len(chunks)))
    summary_table.add_row("Vector store", CHROMA_PERSIST_DIR)
    summary_table.add_row("Time elapsed", f"{elapsed:.1f}s")
    console.print(summary_table)


if __name__ == "__main__":
    main()
