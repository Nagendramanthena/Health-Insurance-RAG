"""
Retrieval Engine for Health Insurance Knowledge Base.

Builds a multi-stage retrieval pipeline:
  1. Hybrid Search — EnsembleRetriever (BM25 + ChromaDB vector search)
  2. Cross-Encoder Reranking — ContextualCompressionRetriever with BGE-Reranker

Usage:
    from retriever import get_retriever
    retriever = get_retriever()
    docs = retriever.invoke("What is my deductible?")
"""

from rich.console import Console

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from graph_retriever import GraphRetriever

from config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    ENSEMBLE_WEIGHTS,
    RETRIEVER_K,
    RERANKER_TOP_N,
    COLLECTION_NAME,
    MIN_RELEVANCE_SCORE,
)

console = Console()


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


def get_retriever(
    ensemble_weights: list[float] | None = None,
    retriever_k: int | None = None,
    reranker_top_n: int | None = None,
    use_reranker: bool = True,
):
    """
    Build and return the full retrieval pipeline.

    Args:
        ensemble_weights: [BM25_weight, Vector_weight]. Defaults to config.
        retriever_k: Number of candidates each retriever fetches. Defaults to config.
        reranker_top_n: Final number of documents after reranking. Defaults to config.
        use_reranker: Whether to apply cross-encoder reranking. Defaults to True.

    Returns:
        A LangChain retriever (EnsembleRetriever or ContextualCompressionRetriever).
    """
    weights = ensemble_weights or ENSEMBLE_WEIGHTS
    k = retriever_k or RETRIEVER_K
    top_n = reranker_top_n or RERANKER_TOP_N

    console.print("\n⚙️  Building retrieval pipeline...", style="bold cyan")

    # ── Step 1: Vector Retriever ────────────────────────────────
    console.print("  📦 Loading ChromaDB vector store...")
    vectorstore = _load_vectorstore()
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    console.print(f"    ✅ Vector retriever ready (k={k})")

    # ── Step 2: BM25 Retriever ──────────────────────────────────
    console.print("  📝 Building BM25 index from stored documents...")
    all_docs = _get_all_documents(vectorstore)
    bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)
    console.print(f"    ✅ BM25 retriever ready (k={k}, {len(all_docs)} documents indexed)")

    # ── Step 3: Ensemble Retriever ──────────────────────────────
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=weights,
    )
    console.print(f"    ✅ Ensemble retriever ready (weights: BM25={weights[0]}, Vector={weights[1]})")

    if not use_reranker:
        console.print("  🎯 Pipeline ready (no reranking)\n")
        return ensemble_retriever

    # ── Step 4: Cross-Encoder Reranking ─────────────────────────
    console.print(f"  🔄 Loading reranker model: [bold]{RERANKER_MODEL}[/]...")
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    
    class ScoredReranker(CrossEncoderReranker):
        def compress_documents(self, documents, query, callbacks=None):
            if not documents: return []
            scores = self.model.score([(query, d.page_content) for d in documents])
            for d, s in zip(documents, scores):
                d.metadata["relevance_score"] = float(s)
            
            # Sort by score and filter by MIN_RELEVANCE_SCORE
            scored = sorted(documents, key=lambda x: x.metadata["relevance_score"], reverse=True)
            filtered = [d for d in scored if d.metadata["relevance_score"] >= MIN_RELEVANCE_SCORE]
            return filtered[:self.top_n]

    compressor = ScoredReranker(model=cross_encoder, top_n=top_n)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )
    console.print(f"    ✅ Reranker ready (top_n={top_n}, min_score={MIN_RELEVANCE_SCORE})")

    # ── Step 5: Knowledge Graph Integration ─────────────────────
    console.print("  🌐 Initializing Knowledge Graph retriever...")
    graph_retriever = GraphRetriever()
    
    class GraphEnhancedRetriever:
        def __init__(self, base_retriever, graph_retriever):
            self.base_retriever = base_retriever
            self.graph_retriever = graph_retriever
            
        def invoke(self, query: str) -> list[Document]:
            # 1. Get graph context (high precision structured data)
            graph_docs = self.graph_retriever.invoke(query)
            
            # 2. Get vector/BM25 chunks
            base_docs = self.base_retriever.invoke(query)
            
            # 3. Combine: Graph context comes first if found
            return graph_docs + base_docs

    final_retriever = GraphEnhancedRetriever(compression_retriever, graph_retriever)
    console.print("  🎯 Full pipeline (Hybrid + Reranker + Graph) ready!\n")

    return final_retriever


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
    all_docs = _get_all_documents(vectorstore)
    return BM25Retriever.from_documents(all_docs, k=k or RERANKER_TOP_N)
