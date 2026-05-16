import threading
from datetime import datetime, timedelta
from typing import Dict, Optional
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.orchestrator import Orchestrator

class SessionManager:
    """
    Thread-safe manager for Orchestrator instances per session.
    Implements a simple TTL-based cleanup (conceptual for now).
    """
    def __init__(self):
        self._sessions: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def get_orchestrator(self, session_id: str) -> Orchestrator:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "orchestrator": Orchestrator(),
                    "last_accessed": datetime.now()
                }
            else:
                self._sessions[session_id]["last_accessed"] = datetime.now()
            return self._sessions[session_id]["orchestrator"]

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
            to_delete = []
            for sid, data in self._sessions.items():
                if now - data["last_accessed"] > timedelta(hours=max_age_hours):
                    to_delete.append(sid)
            for sid in to_delete:
                del self._sessions[sid]

# Global singleton
manager = SessionManager()
