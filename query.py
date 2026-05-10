"""
Interactive Query Tool for Health Insurance Knowledge Base.

Test the retrieval pipeline with natural language health insurance queries.

Usage:
    python query.py                     # Interactive mode
    python query.py --demo              # Run sample queries
    python query.py --compare "query"   # Compare retriever types
"""

import argparse
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box

from retriever import get_retriever, get_vector_only_retriever, get_bm25_only_retriever

console = Console()

# ──────────────────────────────────────────────────────────────
# Sample queries for demo mode
# ──────────────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    "What is my deductible for the Gold plan?",
    "Is metformin covered in the formulary?",
    "How do I submit a claim for reimbursement?",
    "Do I need prior authorization for an MRI?",
    "Who are the in-network cardiologists?",
    "What preventive care services are covered at no cost?",
    "What is the copay for a specialist visit on the Silver plan?",
    "Does the plan cover telehealth visits?",
]


def display_results(query: str, docs: list, elapsed: float, retriever_name: str = "Hybrid + Reranker"):
    """Display retrieved chunks in a formatted table."""
    console.print(
        Panel(
            f"[bold]{query}[/bold]",
            title=f"🔍 Query ({retriever_name})",
            subtitle=f"{len(docs)} results in {elapsed:.2f}s",
            border_style="cyan",
        )
    )

    if not docs:
        console.print("  [yellow]No results found.[/yellow]\n")
        return

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = meta.get("source_file", "Unknown")
        doc_type = meta.get("doc_type", "Unknown").replace("_", " ").title()
        tier = meta.get("plan_tier", "all")
        page = meta.get("page", "N/A")
        row_range = meta.get("row_range", None)

        # Build subtitle
        subtitle_parts = [f"Type: {doc_type}"]
        if tier != "all":
            subtitle_parts.append(f"Plan: {tier}")
        if row_range:
            subtitle_parts.append(f"Rows: {row_range}")
        elif page != "N/A":
            subtitle_parts.append(f"Page: {int(page) + 1}")

        # Relevance score (if available from reranker)
        relevance = meta.get("relevance_score", None)
        if relevance is not None:
            subtitle_parts.append(f"Score: {relevance:.4f}")

        # Truncate content for display
        content = doc.page_content
        if len(content) > 500:
            content = content[:500] + "..."

        # Prepend contextual header if available (stored in metadata)
        display_header = meta.get("display_header", "")
        full_display_content = display_header + content

        console.print(
            Panel(
                full_display_content,
                title=f"[bold green]#{i}[/bold green] — {source}",
                subtitle=" | ".join(subtitle_parts),
                border_style="dim",
                width=100,
            )
        )

    console.print()


def run_query(query: str, retriever, retriever_name: str = "Hybrid + Reranker"):
    """Execute a single query and display results."""
    start = time.time()
    docs = retriever.invoke(query)
    elapsed = time.time() - start
    display_results(query, docs, elapsed, retriever_name)
    return docs


def compare_retrievers(query: str):
    """Compare results from vector-only, BM25-only, and hybrid+reranker."""
    console.print(
        Panel(
            "[bold]Comparing three retrieval strategies[/bold]\n"
            "1. Vector-only (semantic similarity)\n"
            "2. BM25-only (keyword matching)\n"
            "3. Hybrid + Cross-Encoder Reranking (best of both)",
            title="⚡ Retriever Comparison",
            border_style="magenta",
        )
    )

    # Vector only
    console.print("\n[bold yellow]━━━ Vector-Only Retriever ━━━[/bold yellow]")
    vector_ret = get_vector_only_retriever()
    run_query(query, vector_ret, "Vector-Only")

    # BM25 only
    console.print("\n[bold yellow]━━━ BM25-Only Retriever ━━━[/bold yellow]")
    bm25_ret = get_bm25_only_retriever()
    run_query(query, bm25_ret, "BM25-Only")

    # Hybrid + Reranker
    console.print("\n[bold yellow]━━━ Hybrid + Reranker ━━━[/bold yellow]")
    hybrid_ret = get_retriever()
    run_query(query, hybrid_ret, "Hybrid + Reranker")


def interactive_mode():
    """Run interactive query loop."""
    console.print(
        Panel(
            "[bold]Health Insurance Knowledge Base — Interactive Query Tool[/bold]\n\n"
            "Ask any question about your health insurance.\n"
            "Type [bold cyan]'quit'[/bold cyan] to exit, "
            "[bold cyan]'demo'[/bold cyan] for sample queries, or "
            "[bold cyan]'compare <query>'[/bold cyan] to compare retriever types.",
            title="🏥 Health Insurance AI Copilot",
            border_style="magenta",
            width=80,
        )
    )

    retriever = get_retriever()

    while True:
        console.print()
        query = Prompt.ask("[bold cyan]Your question[/bold cyan]")

        if query.lower() in ("quit", "exit", "q"):
            console.print("\n[bold]Goodbye! 👋[/bold]\n")
            break
        elif query.lower() == "demo":
            for q in SAMPLE_QUERIES:
                run_query(q, retriever)
        elif query.lower().startswith("compare "):
            compare_query = query[8:].strip()
            if compare_query:
                compare_retrievers(compare_query)
            else:
                console.print("[yellow]Usage: compare <your query>[/yellow]")
        elif query.strip():
            run_query(query, retriever)
        else:
            console.print("[yellow]Please enter a query.[/yellow]")


def demo_mode():
    """Run all sample queries."""
    console.print(
        Panel(
            "[bold]Running sample health insurance queries[/bold]",
            title="📋 Demo Mode",
            border_style="green",
        )
    )

    retriever = get_retriever()

    for query in SAMPLE_QUERIES:
        run_query(query, retriever)


def main():
    parser = argparse.ArgumentParser(
        description="Query the Health Insurance Knowledge Base"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run sample queries",
    )
    parser.add_argument(
        "--compare",
        type=str,
        metavar="QUERY",
        help="Compare retriever strategies for a given query",
    )
    args = parser.parse_args()

    if args.compare:
        compare_retrievers(args.compare)
    elif args.demo:
        demo_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
