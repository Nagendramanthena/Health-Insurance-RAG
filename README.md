---
title: Health-Insurance-RAG
emoji: 🏥
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 🏥 Health Insurance AI Copilot & RAG Pipeline

A high-fidelity AI copilot designed for answering structured and unstructured health insurance policy queries. Built using FastAPI, Streamlit, LangGraph, Docling layout-aware chunking, NetworkX Knowledge Graphs, and Google Gemini.

## Features
- **Dynamic RAG Pipeline**: Combines BM25 lexical search, dense vector embeddings, and Cross-Encoder reranking.
- **Structured Knowledge Graph (Graph RAG)**: Maps relationships between insurance plans, drugs, conditions, and network providers for complex, multi-hop lookups.
- **High-Fidelity Developer Console**: Interactive UI to inspect and visualize the live agent orchestration node workflows.
- **Unified Sidecar Container**: FASTApi backend and Streamlit frontend served on a unified port (`7860`).
