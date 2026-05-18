"""
Mem0 In-Memory Layer for Health Insurance AI Copilot.

Keeps a lightweight semantic memory PER SESSION, stored entirely in RAM.
- No disk persistence — resets cleanly on server restart / HF Space refresh.
- Namespaced by session_id so sessions never see each other's memories.
- Mem0 auto-extracts key facts (plan tier, drugs, preferences) from the
  conversation so the agent stays context-aware throughout the session.

Each session gets its own Memory instance (created in SessionManager).
This module provides helpers that operate on a given Memory instance.
"""

import os
import sys
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def create_session_memory():
    """
    Create a fresh in-memory Mem0 Memory instance for a single session.
    Uses Qdrant in-memory vector store — no files written, no API key needed.
    Falls back gracefully if mem0 isn't installed or MEM0_ENABLED=false.

    Returns:
        A Mem0 Memory instance, or None if mem0 is unavailable/disabled.
    """
    from config import MEM0_ENABLED, MEM0_LLM_MODEL, MEM0_EMBEDDER_MODEL

    if not MEM0_ENABLED:
        logger.info("ℹ️  Mem0 disabled via MEM0_ENABLED=false")
        return None

    try:
        from mem0 import Memory

        config = {
            # In-memory Qdrant — lives only as long as this Python object
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "session_memories",
                    "in_memory": True,
                },
            },
            # Cheap fast model for fact extraction
            "llm": {
                "provider": "openai",
                "config": {
                    "model": MEM0_LLM_MODEL,
                    "temperature": 0,
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            # Small fast embedder
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": MEM0_EMBEDDER_MODEL,
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        }

        mem = Memory.from_config(config)
        logger.info("✅ Mem0 session memory created (in-memory / ephemeral)")
        return mem

    except ImportError:
        logger.warning("⚠️  mem0ai not installed — memory features disabled")
        return None
    except Exception as e:
        logger.warning(f"⚠️  Mem0 init failed: {e} — memory features disabled")
        return None


# ── Per-session helpers ───────────────────────────────────────────────────────

SESSION_USER = "session"   # Fixed user_id within each isolated Memory instance


def search_memories(mem, query: str, limit: int = 5) -> str:
    """
    Search this session's memory for facts relevant to the current query.

    Args:
        mem:   The session Memory instance (from create_session_memory)
        query: Current user question
        limit: Max facts to return

    Returns:
        Formatted string of facts, or empty string if none found.
    """
    if mem is None:
        return ""

    try:
        results = mem.search(query, user_id=SESSION_USER, limit=limit)
        facts = [r["memory"] for r in results.get("results", [])]

        if not facts:
            return ""

        lines = ["📋 Relevant facts recalled from this session:"]
        for i, fact in enumerate(facts, 1):
            lines.append(f"  {i}. {fact}")
        return "\n".join(lines)

    except Exception as e:
        logger.debug(f"Mem0 search skipped: {e}")
        return ""


def add_memory(mem, query: str, answer: str) -> list:
    """
    Extract and store key facts from this Q&A turn into session memory.

    Args:
        mem:    The session Memory instance
        query:  User's question
        answer: AI's answer

    Returns:
        List of extracted fact strings (for logging).
    """
    if mem is None:
        return []

    try:
        messages = [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": answer},
        ]
        result = mem.add(messages, user_id=SESSION_USER)

        stored = []
        if isinstance(result, dict):
            for item in result.get("results", []):
                if isinstance(item, dict) and "memory" in item:
                    stored.append(item["memory"])
        return stored

    except Exception as e:
        logger.debug(f"Mem0 add skipped: {e}")
        return []


def get_all_memories(mem) -> list:
    """Return all stored facts for the session (for /memory debug endpoint)."""
    if mem is None:
        return []
    try:
        results = mem.get_all(user_id=SESSION_USER)
        return results.get("results", [])
    except Exception as e:
        logger.debug(f"Mem0 get_all skipped: {e}")
        return []
