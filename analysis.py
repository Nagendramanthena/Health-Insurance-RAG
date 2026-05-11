import os
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from rich.console import Console

console = Console()

def print_docling_chunks(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 1. Setup Hybrid Chunker
    # Using the same embedding model as the rest of the project
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    console.print(f"[bold green]Loading tokenizer:[/] {EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=1000, # Can be tuned
        # overlap=200,    # Can be tuned
    )

    # 2. Setup DoclingLoader as requested
    console.print(f"[bold green]Processing file:[/] {os.path.basename(file_path)}")
    loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.DOC_CHUNKS,
        chunker=chunker,
    )

    # 3. Load and print chunks
    chunks = loader.load()
    console.print(chunks[0].metadata["chunk_type"])
    
    # console.print(f"\n[bold yellow]Total chunks found:[/] {len(chunks)}\n")
    
    # for i, chunk in enumerate(chunks):
    #     console.print(f"[bold cyan]--- Chunk {i+1} ---[/]")
    #     # Print a snippet of metadata and the content
    #     # console.print(f"[dim]Metadata:[/] {chunk.metadata}")
    #     console.print(chunk.page_content)
    #     console.print("-" * 40)

if __name__ == "__main__":
    # Test with one of the PDF documents
    sample_pdf = "Documents/SBC_SILVER_SilverShield.pdf"
    print_docling_chunks(sample_pdf)
