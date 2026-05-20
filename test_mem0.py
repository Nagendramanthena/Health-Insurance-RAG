import os
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "session_memories",
            "path": ":memory:",
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
}

try:
    mem = Memory.from_config(config)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
