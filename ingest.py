"""
Document Ingestion Pipeline for Health Insurance Knowledge Base.

Loads PDFs and CSVs from the Documents/ folder, chunks them intelligently,
enriches with metadata and contextual headers, and persists to ChromaDB.
"""

import os
import time
import glob
import shutil
import pandas as pd
from transformers import AutoTokenizer
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_huggingface import HuggingFaceEmbeddings
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
    """Check if the text chunk looks like a Markdown table."""
    return any(line.strip().startswith("|") for line in text.splitlines())


def _build_contextual_header(metadata: dict) -> str:
    """Create a descriptive header for retrieval context."""
    # Note: This header is stored in metadata["display_header"] and 
    # should be prepended to chunks at retrieval time, not during embedding.
    source = metadata.get("source_file", "Unknown")
    doc_type = metadata.get("doc_type", "Unknown").replace("_", " ").title()
    tier = metadata.get("plan_tier", "all")
    
    header = f">>> DOCUMENT CONTEXT <<<\n"
    header += f"Source: {source}\n"
    header += f"Document Type: {doc_type}\n"
    if tier != "all":
        header += f"Plan Tier: {tier}\n"
    
    if "row_range" in metadata:
        header += f"CSV Rows: {metadata['row_range']}\n"
    if "group_value" in metadata:
        header += f"Group: {metadata['group_value']}\n"
        
    header += "------------------------\n"
    return header


# ──────────────────────────────────────────────────────────────
# Document Loading & Chunking
# ──────────────────────────────────────────────────────────────

def load_and_chunk_documents() -> list[Document]:
    """Load and chunk PDFs via Docling and CSVs via specialized Markdown logic."""
    pdf_files = sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf")))
    csv_files = sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.csv")))
    all_chunks = []

    console.print(f"\n⚙️  Configuring Docling with {EMBEDDING_MODEL} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    chunker = HybridChunker(
        tokenizer=tokenizer, 
        max_tokens=MAX_TOKENS_PER_CHUNK,
        overlap=CHUNK_OVERLAP
    )

    # 1. Process PDFs
    if pdf_files:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Processing PDFs...", total=len(pdf_files))
            for file_path in pdf_files:
                filename = os.path.basename(file_path)
                progress.update(task, description=f"Chunking {filename}")
                
                try:
                    # Switch to ExportType.MARKDOWN for better table formatting
                    loader = DoclingLoader(
                        file_path=file_path, 
                        export_type=ExportType.DOC_CHUNKS, # Keeping DOC_CHUNKS to preserve Fix 4 metadata
                        chunker=chunker
                    )
                    chunks = loader.load()
                    
                    classification = _classify_document(filename)
                    for chunk in chunks:
                        # Fix 2: Table detection
                        if _is_table_chunk(chunk.page_content):
                            chunk.metadata["chunk_type"] = "table"
                        else:
                            chunk.metadata["chunk_type"] = "prose"
                            
                        # Fix 1: Store header in metadata, leave content untouched
                        chunk.metadata.update({"source_file": filename, **classification})
                        chunk.metadata["display_header"] = _build_contextual_header(chunk.metadata)
                        
                    all_chunks.extend(chunks)
                except Exception as e:
                    console.print(f"[bold red]Error processing {filename}:[/] {str(e)}")
                
                progress.advance(task)

    # 2. Process CSVs
    if csv_files:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Processing CSVs...", total=len(csv_files))
            for file_path in csv_files:
                filename = os.path.basename(file_path)
                progress.update(task, description=f"Chunking {filename}")
                
                try:
                    classification = _classify_document(filename)
                    df = pd.read_csv(file_path)
                    
                    # Fix 5: Semantic grouping
                    group_col = None
                    for col in df.columns:
                        if any(k in col.lower() for k in ["plan", "tier", "category", "group"]):
                            group_col = col
                            break
                    
                    if group_col:
                        # Group by semantic column and chunk within each group
                        for val, group in df.groupby(group_col):
                            for i in range(0, len(group), CSV_CHUNK_SIZE):
                                batch = group.iloc[i : i + CSV_CHUNK_SIZE]
                                _process_csv_batch(batch, filename, classification, i, val, all_chunks)
                    else:
                        # Fallback to fixed-size slicing
                        for i in range(0, len(df), CSV_CHUNK_SIZE):
                            batch = df.iloc[i : i + CSV_CHUNK_SIZE]
                            _process_csv_batch(batch, filename, classification, i, None, all_chunks)
                except Exception as e:
                    console.print(f"[bold red]Error processing {filename}:[/] {str(e)}")
                
                progress.advance(task)

    console.print(f"  ✅ Created [bold green]{len(all_chunks)}[/] high-precision chunks from [bold]{len(pdf_files) + len(csv_files)}[/] files")
    return all_chunks


def _process_csv_batch(batch, filename, classification, offset, group_val, all_chunks):
    """Helper to format and append CSV chunks."""
    markdown_table = batch.to_markdown(index=False)
    semantic_text = []
    for _, row in batch.iterrows():
        row_desc = " | ".join([f"{k}: {v}" for k, v in row.items() if pd.notna(v)])
        semantic_text.append(row_desc)
    
    content = f"### TABLE DATA (Rows {offset+1}-{offset+len(batch)})\n\n"
    content += markdown_table + "\n\n### ROW DETAILS\n" + "\n".join(semantic_text)
    
    metadata = {
        "source_file": filename,
        "row_range": f"{offset+1}-{offset+len(batch)}",
        "doc_type": classification["doc_type"],
        "plan_tier": classification["plan_tier"],
        "chunk_type": "table"
    }
    if group_val is not None:
        metadata["group_value"] = str(group_val)
        
    # Store header in metadata
    metadata["display_header"] = _build_contextual_header(metadata)
    
    doc = Document(page_content=content, metadata=metadata)
    all_chunks.append(doc)


# ──────────────────────────────────────────────────────────────
# Embedding & Storage
# ──────────────────────────────────────────────────────────────

def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and persist to ChromaDB."""
    # Fix 4: Flatten Docling metadata before it gets filtered
    for chunk in chunks:
        dl_meta = chunk.metadata.get("dl_meta", {})
        if dl_meta:
            chunk.metadata["page_no"] = dl_meta.get("page_no")
            chunk.metadata["heading"] = dl_meta.get("headings", [""])[0]
            # Convert bbox dict to string for Chroma compatibility
            bbox = dl_meta.get("bbox")
            if bbox:
                chunk.metadata["bbox"] = str(bbox)
    
    chunks = filter_complex_metadata(chunks)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Fix 7: Top-level shutil import (done above)
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        console.print("  🗑️  Cleared previous ChromaDB")

    console.print(f"  📦 Embedding [bold]{len(chunks)}[/] chunks and storing in ChromaDB...")

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
    summary_table.add_row("Total layout-aware chunks", str(len(chunks)))
    summary_table.add_row("Vector store location", CHROMA_PERSIST_DIR)
    summary_table.add_row("Time elapsed", f"{elapsed:.1f}s")
    console.print(summary_table)

if __name__ == "__main__":
    main()
