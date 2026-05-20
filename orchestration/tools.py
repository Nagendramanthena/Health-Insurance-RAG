"""
Tools for the Health Insurance AI Copilot Orchestrator.

Defines 4 retrieval tools the LangGraph orchestrator can invoke:
  - policy_search          : General hybrid retrieval (BM25 + Vector + Reranker + Graph)
  - relational_search      : Knowledge Graph structured entity lookup
  - plan_comparison_search : Hybrid retrieval scoped to a specific plan tier
  - prior_auth_search      : Hybrid retrieval filtered to prior-authorization docs

The underlying retrieval pipeline (BM25 + Vector + MultiQuery + CrossEncoder + Graph)
is UNCHANGED — these tools are thin wrappers that control WHAT to query and HOW to filter.
"""

import sys
import os

# Ensure project root is on sys.path so `config` and sibling packages resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langchain_core.documents import Document

from retrieval.retriever import get_hybrid_retriever
from retrieval.graph_retriever import GraphRetriever

# ── Lazy singletons (built once, reused across all tool calls) ──
_hybrid_retriever = None
_graph_retriever  = None


def _get_retrievers():
    global _hybrid_retriever, _graph_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = get_hybrid_retriever()
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _hybrid_retriever, _graph_retriever


def _format_docs(docs: list[Document], max_per_tool: int = 5) -> str:
    """Format retrieved documents with citations."""
    formatted = []
    for d in docs[:max_per_tool]:
        source = d.metadata.get("source_file", "Unknown")
        page   = d.metadata.get("page", "")
        row    = d.metadata.get("row_range", "")
        cite   = f"(Source: {source}"
        if page: cite += f", Page: {int(page)+1}"
        if row:  cite += f", Rows: {row}"
        cite += ")"
        formatted.append(f"{cite}\n{d.page_content}")
    return "\n\n---\n\n".join(formatted) if formatted else ""


# ── Tool 1: General policy search ─────────────────────────────────────────────
@tool
def policy_search(query: str) -> str:
    """
    Search the health insurance policy documents, FAQs, and guidelines.
    Use this for questions about coverage rules, claim procedures, benefit summaries,
    and general insurance terms.
    Uses the full hybrid pipeline: BM25 + Vector + MultiQuery + CrossEncoder Reranker + Graph.
    """
    hybrid, _ = _get_retrievers()
    
    if "claim" in query.lower() or "diagnosis" in query.lower() or "eligib" in query.lower():
        docs = hybrid.invoke(f"claim submission eligibility diagnosis {query}")
    else:
        docs = hybrid.invoke(query)
        
    result = _format_docs(docs)
    return result if result else "No relevant policy information found."


# ── Tool 2: Structured graph lookup ───────────────────────────────────────────
@tool
def relational_search(query: str) -> str:
    """
    Search the knowledge graph for structured relational data.
    Use this for specific lookups like:
    - Copays or coinsurance for a specific drug (e.g., 'What is the copay for Metformin?')
    - Provider details (e.g., 'Which cardiologists are in-network?')
    - Plan-specific relational links between drugs, conditions, and tiers.
    """
    _, graph = _get_retrievers()
    docs = graph.invoke(query)
    return "\n\n---\n\n".join([d.page_content for d in docs]) if docs else "No structured relational data found for this query."


# ── Tool 3: Plan-tier scoped search ───────────────────────────────────────────
@tool
def plan_comparison_search(query: str, tier: str) -> str:
    """
    Search policy documents filtered to a specific plan tier (Bronze, Silver, or Gold).
    Use this when the user wants to compare plans or asks about a specific plan tier.
    Runs the full hybrid retrieval pipeline with a tier-augmented query, then
    prioritises documents whose metadata or content mentions that tier.
    """
    hybrid, _ = _get_retrievers()

    # Augment the query with the specific tier to force relevant documents up the ranking
    tier_query = f"{tier} plan {query}"
    docs = hybrid.invoke(tier_query)

    # Guarantee tier-specific documents (like the SBC deductibles table) are included
    try:
        from retrieval.retriever import _load_vectorstore
        vs = _load_vectorstore()
        tier_specific_docs = vs.similarity_search(query, k=3, filter={"plan_tier": tier.capitalize()})
        docs = tier_specific_docs + docs
    except Exception as e:
        pass

    tier_docs = [
        d for d in docs
        if d.metadata.get("plan_tier", "").lower() in [tier.lower(), "all"]
        or tier.lower() in d.page_content.lower()
        or d.metadata.get("doc_type") == "sbc"
    ]
    if not tier_docs:
        tier_docs = docs

    result = _format_docs(tier_docs, max_per_tool=5)
    return result if result else f"No {tier} plan information found for this query."


# ── Tool 4: Prior-authorization specific search ───────────────────────────────
@tool
def prior_auth_search(query: str) -> str:
    """
    Search specifically for prior authorization requirements.
    Use this when a query asks about pre-approval, step therapy, authorization
    requirements for drugs or procedures, or PA criteria.
    Runs the full hybrid pipeline with a PA-focused query and filters to
    documents tagged as prior_authorization type or containing PA keywords.
    """
    hybrid, _ = _get_retrievers()

    auth_query = f"prior authorization {query}"
    docs = hybrid.invoke(auth_query)

    pa_keywords = {"prior auth", "authorization required", "step therapy",
                   "pre-approval", "pre-authorization", "pa required", "preauth"}

    auth_docs = [
        d for d in docs
        if d.metadata.get("doc_type") == "prior_authorization"
        or any(kw in d.page_content.lower() for kw in pa_keywords)
    ]

    if not auth_docs:
        return "No prior authorization information found for this query."

    result = _format_docs(auth_docs, max_per_tool=3)
    return result if result else "No prior authorization information found for this query."


# ── Expose all tools ───────────────────────────────────────────────────────────
def get_tools():
    return [policy_search, relational_search, plan_comparison_search, prior_auth_search]

def preload_retrievers():
    """Explicitly trigger retriever build (call this on app startup)."""
    _get_retrievers()
