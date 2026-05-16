"""
Central configuration for the Health Insurance RAG Knowledge Base.
All tunable parameters live here.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "storage", "chroma_db")
GRAPH_DATA_PATH = os.path.join(BASE_DIR, "storage", "knowledge_graph.graphml")

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

# ──────────────────────────────────────────────
# Orchestrator & LLM settings
# ──────────────────────────────────────────────
LLM_MODEL = "gpt-4o"            # Used for synthesis (high accuracy)
CLASSIFIER_LLM_MODEL = "gpt-4o-mini"  # Used for intent classification (fast & cheap)
LLM_TEMPERATURE = 0.0

SYSTEM_PROMPT = """You are a helpful and precise Health Insurance AI Copilot. 
Your goal is to answer questions about health insurance plans, coverage, providers, and drug formularies using the provided tools.

GUIDELINES:
1. **Accuracy**: Only answer based on the context retrieved from tools. If the information is not available, say you don't know.
2. **Citations**: Always cite your sources. Use the 'source_file', 'page', or 'row_range' from the metadata.
   Example: "Your deductible is $500 (Source: SBC_SILVER_SilverShield.pdf, Page 2)."
3. **Safety**: 
   - NEVER provide medical advice or diagnosis. 
   - If asked for medical advice, state: "I am an insurance assistant and cannot provide medical advice. Please consult a healthcare professional."
   - Protect PHI/PII. Do not ask for or store social security numbers or sensitive personal health details.
4. **Tone**: Be professional, clear, and empathetic.
5. **Tool Usage**:
   - Use 'policy_search' for general coverage rules, FAQs, and procedures.
   - Use 'relational_search' for specific data like copays for a drug, provider lookups, or plan-specific relational details.
"""
