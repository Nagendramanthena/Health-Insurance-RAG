import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# UI Aesthetics
APP_TITLE = "HealthGuard AI Copilot"
APP_SUBTITLE = "Your Intelligent Health Insurance Assistant"

# Glassmorphism Theme Colors
THEME = {
    "primary": "#4F46E5",  # Indigo
    "secondary": "#10B981", # Emerald
    "background": "#0F172A", # Slate 900
    "card_bg": "rgba(30, 41, 59, 0.7)", # Slate 800 with opacity
    "text": "#F8FAFC",
    "text_dim": "#94A3B8"
}

# Example Queries
EXAMPLE_QUERIES = [
    "What is the copay for Metformin on the Silver plan?",
    "Which cardiologists are in-network in New York?",
    "How do I file a claim for an out-of-network provider?",
    "Does my plan require prior authorization for physical therapy?",
    "Compare the deductibles for Bronze and Gold plans."
]

# Intent Labels
INTENT_METADATA = {
    "SIMPLE_LOOKUP": {"color": "blue", "icon": "🔍"},
    "POLICY_QUESTION": {"color": "green", "icon": "📄"},
    "MULTI_HOP": {"color": "purple", "icon": "🔗"},
    "COMPARISON": {"color": "orange", "icon": "⚖️"}
}
