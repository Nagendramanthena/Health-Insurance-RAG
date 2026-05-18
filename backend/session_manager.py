import threading
from datetime import datetime, timedelta
from typing import Dict, Optional
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.orchestrator import Orchestrator
from orchestration.memory import create_session_memory

class SessionManager:
    """
    Thread-safe manager for Orchestrator instances per session.

    Each session gets its own:
      - Orchestrator  (LangGraph pipeline)
      - Memory        (Mem0 in-memory — ephemeral, resets on server restart)

    Memory is namespaced to the session — users never see each other's facts.
    When Hugging Face refreshes the Space, all in-memory data is cleared (expected).
    """
    def __init__(self):
        self._sessions: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def get_orchestrator(self, session_id: str) -> Orchestrator:
        with self._lock:
            if session_id not in self._sessions:
                # Create in-memory Mem0 instance for this session
                mem = create_session_memory()
                self._sessions[session_id] = {
                    "orchestrator":   Orchestrator(user_id=session_id, mem=mem),
                    "mem":            mem,
                    "last_accessed":  datetime.now()
                }
            else:
                self._sessions[session_id]["last_accessed"] = datetime.now()
            return self._sessions[session_id]["orchestrator"]

    def get_memory(self, session_id: str):
        """Return the Mem0 memory instance for a session (for /memory debug endpoint)."""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id].get("mem")
            return None

    def clear_session(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def get_active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Removes sessions older than max_age_hours."""
        with self._lock:
            now = datetime.now()
            to_delete = [
                sid for sid, data in self._sessions.items()
                if now - data["last_accessed"] > timedelta(hours=max_age_hours)
            ]
            for sid in to_delete:
                del self._sessions[sid]

# Global singleton
manager = SessionManager()
