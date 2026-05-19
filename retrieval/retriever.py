"""
Retrieval Engine for Health Insurance Knowledge Base.

Builds a multi-stage retrieval pipeline:
  1. Hybrid Search    — EnsembleRetriever (BM25 + ChromaDB vector search)
  2. Multi-Query      — 3 query rephrases via gpt-4o-mini for better recall
  3. Cross-Encoder    — BGE Reranker filters to top-N most relevant chunks
  4. Knowledge Graph  — Structured entity facts prepended for precision

Usage:
    from retrieval.retriever import get_hybrid_retriever
    retriever = get_hybrid_retriever()
    docs = retriever.invoke("What is my deductible?")
"""

import sys
import os

# Ensure project root is on sys.path so `config` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from retrieval.graph_retriever import GraphRetriever
from orchestration.tracing import log_event

from config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    ENSEMBLE_WEIGHTS,
    RETRIEVER_K,
    RERANKER_TOP_N,
    COLLECTION_NAME,
    MIN_RELEVANCE_SCORE,
    LLM_MODEL,
    CLASSIFIER_LLM_MODEL,  # gpt-4o-mini — used for cheap MultiQuery rephrasing
    LLM_TEMPERATURE,
)

console = Console()


# ══════════════════════════════════════════════════════════════
# Multi-Query Retriever
# ══════════════════════════════════════════════════════════════

class SimpleMultiQueryRetriever(Runnable):
    """
    Generate multiple query variants via an LLM and aggregate results.

    Uses gpt-4o-mini (cheap + fast) to rephrase the query 3 ways,
    runs the base retriever for each variant, and deduplicates results.
    This significantly improves document recall without touching precision
    (the reranker handles precision downstream).
    """

    def __init__(self, base_retriever, llm, num_variants: int = 3):
        self.base_retriever = base_retriever
        self.llm = llm
        self.num_variants = num_variants

    def _generate_variants(self, query: str) -> list[str]:
        prompt = (
            f"You are a medical search expert. Generate {self.num_variants} different search queries "
            f"to retrieve relevant documents for: '{query}'.\n"
            "Include variations using synonyms for conditions, symptoms, or drug classes (e.g., 'headache' -> 'pain', 'acid reflux' -> 'GERD').\n"
            "Return ONLY the queries, one per line."
        )
        response = self.llm.invoke(prompt)
        variants = [line.strip() for line in response.content.splitlines() if line.strip()]
        if len(variants) < self.num_variants:
            variants = [query] * self.num_variants
        
        # Log variants for the Developer Console
        for i, v in enumerate(variants):
            log_event(f"🧠 Multi-Query Variant {i+1}: {v}")
            
        return variants

    def invoke(self, query: str, **kwargs) -> list[Document]:
        all_docs: list[Document] = []
        seen_keys = set()
        for variant in self._generate_variants(query):
            docs = self.base_retriever.invoke(variant)
            for d in docs:
                key = (
                    d.metadata.get("source_file"),
                    d.metadata.get("page"),
                    d.page_content[:30],
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_docs.append(d)
        return all_docs


# ══════════════════════════════════════════════════════════════
# Vector Store Helpers
# ══════════════════════════════════════════════════════════════

def _load_vectorstore() -> Chroma:
    """Load the persisted ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectorstore


def _get_all_documents(vectorstore: Chroma) -> list[Document]:
    """
    Extract all documents from ChromaDB to build the BM25 index.
    BM25Retriever needs the full document list in memory.
    """
    collection = vectorstore._collection
    results = collection.get(include=["documents", "metadatas"])

    docs = []
    for content, metadata in zip(results["documents"], results["metadatas"]):
        docs.append(Document(page_content=content, metadata=metadata or {}))

    return docs


# ══════════════════════════════════════════════════════════════
# Main Pipeline Builder
# ══════════════════════════════════════════════════════════════

def get_hybrid_retriever(
    ensemble_weights: list[float] | None = None,
    retriever_k: int | None = None,
    reranker_top_n: int | None = None,
    use_reranker: bool = True,
):
    """
    Build and return the full hybrid retrieval pipeline.

    Pipeline stages:
      1. Vector Retriever   — ChromaDB semantic similarity (k candidates)
      2. BM25 Retriever     — Keyword frequency search (k candidates)
      3. Ensemble Retriever — Weighted merge of BM25 + Vector results
      4. MultiQuery         — 3 rephrases via gpt-4o-mini → broader recall
      5. CrossEncoder       — BGE reranker scores & filters to top_n
      6. Graph Prepend      — Knowledge graph entity facts added first

    Args:
        ensemble_weights: [BM25_weight, Vector_weight]. Defaults to config.
        retriever_k:      Number of candidates each retriever fetches.
        reranker_top_n:   Final number of documents after reranking.
        use_reranker:     Whether to apply cross-encoder reranking.

    Returns:
        A retriever with a `.invoke(query)` method.
    """
    weights = ensemble_weights or ENSEMBLE_WEIGHTS
    k       = retriever_k or RETRIEVER_K
    top_n   = reranker_top_n or RERANKER_TOP_N

    console.print("\n⚙️  Building retrieval pipeline...", style="bold cyan")

    # ── Stage 1: Vector Retriever ───────────────────────────────
    console.print("  📦 Loading ChromaDB vector store...")
    vectorstore = _load_vectorstore()
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    console.print(f"    ✅ Vector retriever ready (k={k})")

    # ── Stage 2: BM25 Retriever ─────────────────────────────────
    console.print("  📝 Building BM25 index from stored documents...")
    all_docs = _get_all_documents(vectorstore)
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)
    console.print(f"    ✅ BM25 retriever ready (k={k}, {len(all_docs)} documents indexed)")

    # ── Stage 3: Ensemble Retriever ─────────────────────────────
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=weights,
    )
    console.print(f"    ✅ Ensemble retriever ready (weights: BM25={weights[0]}, Vector={weights[1]})")

    # ── Stage 4: Knowledge Graph Integration ────────────────────
    # Moved before Multi-Query so variants can be used for graph lookups too.
    console.print("  🌐 Initializing Knowledge Graph retriever...")
    graph_retriever = GraphRetriever()

    class GraphEnhancedRetriever:
        def __init__(self, base_retriever, graph_retriever):
            self.base_retriever   = base_retriever
            self.graph_retriever  = graph_retriever

        def invoke(self, query: str) -> list[Document]:
            # Graph context first (structured, high-precision)
            graph_docs = self.graph_retriever.invoke(query)
            # Hybrid pipeline docs
            base_docs  = self.base_retriever.invoke(query)
            return graph_docs + base_docs

    graph_hybrid_retriever = GraphEnhancedRetriever(ensemble_retriever, graph_retriever)

    # ── Stage 5: Multi-Query Retriever ──────────────────────────
    # Uses gpt-4o-mini — rephrasing a query is simple, no need for gpt-4o.
    console.print("  🧠 Initializing Multi-Query generation (gpt-4o-mini)...")
    llm = ChatOpenAI(temperature=LLM_TEMPERATURE, model=CLASSIFIER_LLM_MODEL)
    multi_query_retriever = SimpleMultiQueryRetriever(
        base_retriever=graph_hybrid_retriever, llm=llm, num_variants=3
    )
    console.print("    ✅ Multi-Query retriever ready")

    if not use_reranker:
        console.print("  🎯 Pipeline ready (no reranking)\n")
        return multi_query_retriever

    # ── Stage 6: Cross-Encoder Reranking ────────────────────────
    console.print(f"  🔄 Loading reranker model: [bold]{RERANKER_MODEL}[/]...")
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)

    class ScoredReranker(CrossEncoderReranker):
        def compress_documents(self, documents, query, callbacks=None):
            if not documents:
                return []
            scores = self.model.score([(query, d.page_content) for d in documents])
            for d, s in zip(documents, scores):
                d.metadata["relevance_score"] = float(s)

            scored   = sorted(documents, key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
            filtered = [
                d for d in scored 
                if d.metadata.get("doc_type") == "knowledge_graph" 
                or d.metadata.get("relevance_score", 0) >= MIN_RELEVANCE_SCORE
            ]
            return filtered[:self.top_n]

    compressor = ScoredReranker(model=cross_encoder, top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever,
    )
    console.print(f"    ✅ Reranker ready (top_n={top_n}, min_score={MIN_RELEVANCE_SCORE})")
    console.print("  🎯 Full pipeline (Hybrid + Reranker + Graph) ready!\n")

    return compression_retriever


# ══════════════════════════════════════════════════════════════
# Comparison Retrievers (for testing only)
# ══════════════════════════════════════════════════════════════

def get_vector_only_retriever(k: int | None = None):
    """Get a simple vector-only retriever (for comparison testing)."""
    vectorstore = _load_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or RERANKER_TOP_N},
    )


def get_bm25_only_retriever(k: int | None = None):
    """Get a simple BM25-only retriever (for comparison testing)."""
    vectorstore = _load_vectorstore()
    all_docs    = _get_all_documents(vectorstore)
    return BM25Retriever.from_documents(all_docs, k=k or RERANKER_TOP_N)
