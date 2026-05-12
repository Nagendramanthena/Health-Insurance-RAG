"""
Tools for the Health Insurance AI Copilot Orchestrator.
Defines retrieval interfaces that the agent can call.
"""

from langchain_core.tools import tool
from retriever import get_hybrid_retriever
from graph_retriever import GraphRetriever
from typing import List
from langchain_core.documents import Document

# Initialize retrievers once
_hybrid_retriever = None
_graph_retriever = None

def _get_retrievers():
    global _hybrid_retriever, _graph_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = get_hybrid_retriever()
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _hybrid_retriever, _graph_retriever

@tool
def policy_search(query: str) -> str:
    """
    Search the health insurance policy documents, FAQs, and guidelines.
    Use this for questions about coverage rules, claim procedures, benefit summaries, 
    and general insurance terms.
    """
    hybrid, _ = _get_retrievers()
    docs = hybrid.invoke(query)
    
    formatted_docs = []
    for d in docs:
        source = d.metadata.get("source_file", "Unknown")
        page = d.metadata.get("page", "")
        row = d.metadata.get("row_range", "")
        cite = f"(Source: {source}"
        if page: cite += f", Page: {int(page)+1}"
        if row: cite += f", Rows: {row}"
        cite += ")"
        
        formatted_docs.append(f"{cite}\n{d.page_content}")
        
    return "\n\n---\n\n".join(formatted_docs) if formatted_docs else "No relevant policy information found."

@tool
def relational_search(query: str) -> str:
    """
    Search the knowledge graph for structured relational data.
    Use this for specific lookups like:
    - Copays or coinsurance for a specific drug (e.g., 'What is the copay for Metformin?')
    - Provider details (e.g., 'Which cardiologists are in-network?')
    - Plan-specific relational links.
    """
    _, graph = _get_retrievers()
    docs = graph.invoke(query)
    
    return "\n\n---\n\n".join([d.page_content for d in docs]) if docs else "No structured relational data found for this query."

def get_tools():
    return [policy_search, relational_search]
