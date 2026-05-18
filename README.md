---
title: Health-Insurance-RAG
emoji: 🏥
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🏥 Health Insurance AI Copilot & RAG Pipeline

A high-fidelity AI copilot for answering structured and unstructured health insurance policy queries. Built with FastAPI, Streamlit, LangGraph, Mem0 agent memory, Docling layout-aware chunking, NetworkX Knowledge Graphs, and OpenAI GPT-4o.

## Features
- **Dynamic RAG Pipeline**: Combines BM25 lexical search, dense vector embeddings, and Cross-Encoder reranking.
- **Structured Knowledge Graph (Graph RAG)**: Maps relationships between insurance plans, drugs, conditions, and network providers for complex, multi-hop lookups.
- **🧠 Mem0 Session Memory**: Agent remembers facts within a session (plan tier, drug preferences, prior auth history) for personalized, context-aware answers — resets cleanly on Space refresh.
- **5-Node LangGraph Orchestrator**: `memory_search → classify_intent → retrieve → synthesize → memory_add`
- **High-Fidelity Developer Console**: Interactive UI to inspect and visualize live agent orchestration node workflows.
- **Unified Sidecar Container**: FastAPI backend and Streamlit frontend served on a single port (`7860`).

## Deployment on Hugging Face Spaces

1. Push this repo to your HF Space (Docker SDK, port 7860)
2. Go to **Settings → Variables and Secrets** and add:
   - `OPENAI_API_KEY` → your OpenAI key
   - `GOOGLE_API_KEY` → your Google key (optional)
3. The Space will build automatically — Mem0 runs in-memory, no extra setup needed.

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a query, get answer + memory facts used |
| `GET` | `/memory/{session_id}` | Inspect what Mem0 remembers for a session |
| `DELETE` | `/memory/{session_id}` | Clear session memory (keep chat history) |
| `DELETE` | `/session/{session_id}` | Clear full session including memory |
| `GET` | `/health` | Health check |
| `GET` | `/dev-console` | Live developer console |
