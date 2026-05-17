#!/bin/bash
# Start FastAPI Backend
echo "🚀 Starting Health Insurance RAG Backend..."
export PYTHONPATH=$PYTHONPATH:.
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
