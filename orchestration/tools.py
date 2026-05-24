"""
Tools for the Health Insurance AI Copilot Orchestrator.

Defines 4 retrieval tools the LangGraph orchestrator can invoke:
  - policy_search          : General hybrid retrieval (BM25 + Vector + Reranker + Graph)
  - relational_search      : Knowledge Graph structured entity lookup
  - plan_comparison_search : Hybrid retrieval scoped to a specific plan tier
  - prior_auth_search      : Hybrid retrieval filtered to prior-authorization docs

ENHANCEMENTS:
  - Doc-type routing: pre-routes queries to the most relevant document collection
  - Metadata pre-filtering: scopes vector searches to relevant doc types
  - Per-tool query augmentation for higher precision
"""

import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langchain_core.documents import Document

from retrieval.retriever import get_hybrid_retriever
from retrieval.graph_retriever import GraphRetriever

# ── Lazy singletons ────────────────────────────────────────────────────────────
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
        # Support both 'page' (0-indexed) and 'page_no' (1-indexed) metadata fields
        page = d.metadata.get("page")
        page_no = d.metadata.get("page_no")
        row = d.metadata.get("row_range", "")
        cite = f"(Source: {source}"
        if page is not None and str(page).strip() != "":
            try:
                # If page is stored, it's 0-indexed, display as 1-indexed
                cite += f", Page: {int(page)+1}"
            except ValueError:
                cite += f", Page: {page}"
        elif page_no is not None and str(page_no).strip() != "":
            # page_no is 1-indexed directly from Docling
            cite += f", Page: {page_no}"
            
        if row:  cite += f", Rows: {row}"
        cite += ")"
        
        # Prepend contextual display header (plan tier, document type, source)
        header = d.metadata.get("display_header", "")
        content = f"{header}\n{d.page_content}" if header else d.page_content
        formatted.append(f"{cite}\n{content}")
        
    return "\n\n---\n\n".join(formatted) if formatted else ""


# ── Doc-Type Router ────────────────────────────────────────────────────────────
# Maps keyword signals → document collection types for pre-filtering
_DOC_TYPE_KEYWORDS: dict[str, list[str]] = {
    "drug_formulary": [
        "drug", "formulary", "medication", "prescription", "generic", "brand",
        "metformin", "lisinopril", "atorvastatin", "tier 1", "tier 2", "tier 3",
        "covered drug", "formulary list", "pharmacy benefit",
    ],
    "claim_submission_guidelines": [
        "claim", "submit a claim", "reimburs", "billing", "appeal", "dispute",
        "out-of-pocket claim", "claim form", "eoob", "explanation of benefits",
    ],
    "preventive_care_schedule": [
        "preventive", "annual exam", "wellness visit", "vaccine", "immunization",
        "mammogram", "colonoscopy", "screening", "checkup", "physical exam",
    ],
    "prior_authorization": [
        "prior auth", "prior authorization", "step therapy", "pre-approval",
        "preauthorization", "pa required", "approval required", "step-therapy",
    ],
    "provider_directory": [
        "provider", "doctor", "specialist", "in-network", "out-of-network",
        "hospital", "physician", "clinic", "dermatologist", "cardiologist",
        "network provider", "find a doctor",
    ],
    "evidence_of_coverage": [
        "coverage", "covered service", "benefit", "exclusion", "limitation",
        "deductible", "copay", "coinsurance", "out-of-pocket maximum",
        "emergency", "urgent care", "what is covered",
    ],
}

def _infer_doc_type(query: str) -> Optional[str]:
    """Infer the most relevant document type from the query text."""
    q_lower = query.lower()
    best_doc_type = None
    best_match_count = 0

    for doc_type, keywords in _DOC_TYPE_KEYWORDS.items():
        match_count = sum(1 for kw in keywords if kw in q_lower)
        if match_count > best_match_count:
            best_match_count = match_count
            best_doc_type = doc_type

    # Only return a type if we have at least one strong signal
    return best_doc_type if best_match_count >= 1 else None


def _get_doc_type_filtered_docs(vs, query: str, doc_type: str, k: int = 3) -> list[Document]:
    """Fetch docs pre-filtered by doc_type metadata from ChromaDB."""
    try:
        return vs.similarity_search(query, k=k, filter={"doc_type": doc_type})
    except Exception:
        return []


# ── Tool 1: General policy search ─────────────────────────────────────────────
@tool
def policy_search(query: str) -> str:
    """
    Search the health insurance policy documents, FAQs, and guidelines.
    Use this for questions about coverage rules, claim procedures, benefit summaries,
    and general insurance terms.
    Uses the full hybrid pipeline: BM25 + Vector + MultiQuery + CrossEncoder Reranker + Graph.
    Enhanced with doc-type pre-filtering for higher precision.
    """
    hybrid, _ = _get_retrievers()

    # Augment query for claim/eligibility-related searches
    search_query = query
    if any(kw in query.lower() for kw in ["claim", "diagnosis", "eligib"]):
        search_query = f"claim submission eligibility diagnosis {query}"

    docs = hybrid.invoke(search_query)

    # Boost with doc-type pre-filtered results if a strong signal is found
    inferred_type = _infer_doc_type(query)
    if inferred_type:
        try:
            from retrieval.retriever import _load_vectorstore
            vs = _load_vectorstore()
            type_docs = _get_doc_type_filtered_docs(vs, query, inferred_type, k=3)
            if type_docs:
                # Prepend targeted docs — they're likely more relevant
                seen = {d.page_content[:50] for d in docs}
                new_type_docs = [d for d in type_docs if d.page_content[:50] not in seen]
                docs = new_type_docs + docs
        except Exception:
            pass  # Graceful fallback to standard retrieval

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
    Enhanced: uses ChromaDB metadata pre-filter for plan_tier before hybrid search.
    """
    hybrid, _ = _get_retrievers()

    # Tier-augmented query forces relevant docs up the ranking
    tier_query = f"{tier} plan {query}"
    docs = hybrid.invoke(tier_query)

    # Guarantee tier-specific documents via metadata pre-filter (high precision boost)
    try:
        from retrieval.retriever import _load_vectorstore
        vs = _load_vectorstore()

        # Pre-filter by both plan_tier AND inferred doc_type for maximum precision
        tier_specific_docs = vs.similarity_search(query, k=8, filter={"plan_tier": tier.capitalize()})
        docs = tier_specific_docs + docs
    except Exception:
        pass

    # Filter to tier-relevant docs
    tier_docs = [
        d for d in docs
        if d.metadata.get("plan_tier", "").lower() in [tier.lower(), "all"]
        or tier.lower() in d.page_content.lower()
        or d.metadata.get("doc_type") in ("sbc", f"summary_of_benefits_{tier.lower()}")
    ]
    if not tier_docs:
        tier_docs = docs  # Fallback: use all results

    result = _format_docs(tier_docs, max_per_tool=8)
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
    Enhanced: uses ChromaDB metadata pre-filter for prior_authorization docs.
    """
    hybrid, _ = _get_retrievers()

    auth_query = f"prior authorization requirements {query}"
    docs = hybrid.invoke(auth_query)

    # Boost with metadata-filtered PA docs
    try:
        from retrieval.retriever import _load_vectorstore
        vs = _load_vectorstore()
        pa_docs = _get_doc_type_filtered_docs(vs, query, "prior_authorization", k=3)
        if pa_docs:
            seen = {d.page_content[:50] for d in docs}
            new_pa = [d for d in pa_docs if d.page_content[:50] not in seen]
            docs = new_pa + docs
    except Exception:
        pass

    pa_keywords = {
        "prior auth", "authorization required", "step therapy",
        "pre-approval", "pre-authorization", "pa required", "preauth",
        "prior authorization", "requires authorization",
    }

    auth_docs = [
        d for d in docs
        if d.metadata.get("doc_type") in ("prior_authorization",)
        or any(kw in d.page_content.lower() for kw in pa_keywords)
    ]

    if not auth_docs:
        return "No prior authorization information found for this query."

    result = _format_docs(auth_docs, max_per_tool=4)
    return result if result else "No prior authorization information found for this query."


# ── Expose all tools ───────────────────────────────────────────────────────────
def get_tools():
    return [policy_search, relational_search, plan_comparison_search, prior_auth_search]

def preload_retrievers():
    """Explicitly trigger retriever build (call this on app startup)."""
    _get_retrievers()
