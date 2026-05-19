"""
LangSmith integration utilities for the Health Insurance RAG pipeline.

Provides:
  - tag_current_run()  : attach rich metadata (session_id, intent, language)
                         to the currently active LangSmith trace.
  - get_run_id()       : retrieve the UUID of the current LangSmith run,
                         useful for linking API responses to their traces.

Usage (inside any @traceable function or LangGraph node):

    from orchestration.langsmith_tracing import tag_current_run, get_run_id

    tag_current_run(session_id="abc", intent="SIMPLE_LOOKUP", language="English")
    run_id = get_run_id()   # e.g. "3f4a6b12-..." or None if tracing off

All functions are safe no-ops when LANGCHAIN_TRACING_V2 is not set or is false.
"""

import os
import logging

logger = logging.getLogger(__name__)


def _is_tracing_enabled() -> bool:
    """
    Check at call-time whether LangSmith tracing is active.

    Using a function (not a module-level constant) is critical: if this module
    is imported before load_dotenv() runs, a constant would always read False.
    A function reads os.environ fresh on every call, so it correctly reflects
    the state after dotenv has loaded the .env file.
    """
    return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def tag_current_run(
    session_id: str = "default",
    intent: str = "",
    language: str = "English",
    extra_metadata: dict | None = None,
) -> None:
    """
    Attach structured metadata + tags to the currently active LangSmith run.

    Must be called from within a @traceable function or a LangGraph node that
    is being invoked as part of a traced chain. Safe no-op otherwise.

    Args:
        session_id:      The user/session identifier (for filtering in dashboard).
        intent:          The classified query intent (SIMPLE_LOOKUP, etc.).
        language:        The user's detected language (for i18n analytics).
        extra_metadata:  Any additional key-value pairs to attach to the run.
    """
    if not _is_tracing_enabled():
        return

    try:
        from langsmith.run_helpers import get_current_run_tree  # lazy import

        run = get_current_run_tree()
        if run is None:
            return

        metadata: dict = {
            "session_id": session_id,
            "intent": intent,
            "language": language,
            "project": "health-insurance-rag",
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        run.add_metadata(metadata)
        run.add_tags([
            f"intent:{intent}" if intent else "intent:unknown",
            f"lang:{language.lower()}",
            "health-insurance-rag",
        ])

        logger.debug(f"LangSmith run tagged — intent={intent}, lang={language}, session={session_id}")

    except ImportError:
        logger.warning("langsmith not installed. Run: pip install langsmith")
    except Exception as e:
        # Never let tracing failures crash the pipeline.
        logger.debug(f"LangSmith tag_current_run skipped: {e}")


def get_run_id() -> str | None:
    """
    Return the UUID string of the current LangSmith run (top-level trace).

    Call this from inside a @traceable function to capture the run ID
    *before* the function returns (after it returns, the span is closed).

    Returns:
        A UUID string like "3f4a6b12-8c1d-4e9f-a2b0-..." or None.
    """
    if not _is_tracing_enabled():
        return None

    try:
        from langsmith.run_helpers import get_current_run_tree  # lazy import

        run = get_current_run_tree()
        if run is not None:
            return str(run.id)

    except ImportError:
        logger.warning("langsmith not installed. Run: pip install langsmith")
    except Exception as e:
        logger.debug(f"LangSmith get_run_id skipped: {e}")

    return None
