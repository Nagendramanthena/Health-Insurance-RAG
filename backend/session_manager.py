import threading
import json
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.orchestrator import Orchestrator
from backend.database import save_message, get_session_history

class SessionManager:
    """
    Enhanced Session Manager with Redis caching and SQL persistence.
    """
    def __init__(self):
        self._orchestrators: Dict[str, Orchestrator] = {}
        self._lock = threading.Lock()
        
        # Redis setup with fallback
        try:
            self.redis = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis.ping()
            print("✅ Redis connected")
        except:
            self.redis = None
            print("⚠️ Redis not found, using in-memory fallback")

    def get_orchestrator(self, session_id: str) -> Orchestrator:
        with self._lock:
            if session_id not in self._orchestrators:
                orch = Orchestrator()
                # Load history from Database
                history = get_session_history(session_id)
                if history:
                    print(f"📜 Loaded {len(history)} messages from DB for session {session_id}")
                    # Update orchestrator state with history (if your orchestrator supports it)
                    # For now, we'll just keep it in memory
                    pass 
                
                self._orchestrators[session_id] = orch
            
            return self._orchestrators[session_id]

    def add_message(self, session_id: str, role: str, content: str, intent: str = None):
        """Persist message to Database and Cache."""
        # 1. Save to SQL Database (Permanent)
        save_message(session_id, role, content, intent)
        
        # 2. Save to Redis (Fast Access for UI)
        if self.redis:
            msg = {"role": role, "content": content, "intent": intent, "ts": datetime.now().isoformat()}
            self.redis.rpush(f"chat:{session_id}", json.dumps(msg))
            self.redis.expire(f"chat:{session_id}", 86400) # 24h cache

    def get_history(self, session_id: str) -> List[Dict]:
        """Get history from Redis (fast) or DB (fallback)."""
        if self.redis:
            cached = self.redis.lrange(f"chat:{session_id}", 0, -1)
            if cached:
                return [json.loads(m) for m in cached]
        
        # Fallback to Database
        return get_session_history(session_id)

    def clear_session(self, session_id: str):
        with self._lock:
            if session_id in self._orchestrators:
                del self._orchestrators[session_id]
        if self.redis:
            self.redis.delete(f"chat:{session_id}")

# Global singleton
manager = SessionManager()
