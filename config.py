"""
Central configuration for the Health Insurance RAG Knowledge Base.
All tunable parameters live here.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "Documents")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
GRAPH_DATA_PATH = os.path.join(BASE_DIR, "knowledge_graph.graphml")

# ──────────────────────────────────────────────
# Document type classification (by filename substring)
# ──────────────────────────────────────────────
DOC_TYPE_MAP = {
    "EOC": "evidence_of_coverage",
    "SBC_BRONZE": "summary_of_benefits_bronze",
    "SBC_SILVER": "summary_of_benefits_silver",
    "SBC_GOLD": "summary_of_benefits_gold",
    "ClaimSubmission": "claim_submission_guidelines",
    "MemberFAQ": "member_faq_glossary",
    "PreventiveCare": "preventive_care_schedule",
    "PriorAuthorization": "prior_authorization",
    "Drug_Formulary": "drug_formulary",
    "InNetwork_Provider": "provider_directory",
}

PLAN_TIER_MAP = {
    "SBC_BRONZE": "Bronze",
    "SBC_SILVER": "Silver",
    "SBC_GOLD": "Gold",
}

# ──────────────────────────────────────────────
# Chunking parameters
# ──────────────────────────────────────────────
MAX_TOKENS_PER_CHUNK = 384   # tokens per chunk for Docling's HybridChunker
CHUNK_OVERLAP = int(MAX_TOKENS_PER_CHUNK * 0.15)
CSV_CHUNK_SIZE = 10         # rows per chunk for CSV files (better for context)

# ──────────────────────────────────────────────
# Embedding model
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# Reranker model
# ──────────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-base"

# ──────────────────────────────────────────────
# Retrieval parameters
# ──────────────────────────────────────────────
# EnsembleRetriever weights: [BM25_weight, Vector_weight]
ENSEMBLE_WEIGHTS = [0.4, 0.6]

# How many candidates each individual retriever fetches
RETRIEVER_K = 20

# Final number of chunks after reranking
RERANKER_TOP_N = 5

# Minimum relevance score from Cross-Encoder to consider a result valid
# Results below this threshold will be filtered out.
MIN_RELEVANCE_SCORE = 0.05 

# ChromaDB collection name
COLLECTION_NAME = "health_insurance_kb"

RECORD_CSV_MIN_ROWS = 200   # CSVs with more rows than this are candidates for record mode
RECORD_CSV_MIN_COLS = 8     # ...and where each row has at least this many non-null fields
