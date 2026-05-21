"""
Health Insurance AI Copilot — LangGraph Sequential Chain Orchestrator.

Architecture (5 nodes in a directed StateGraph):

    START
      │
      ▼
  [1. memory_search]  ← NEW: Retrieve relevant past facts from Mem0
      │
      ▼
  [2. classify_intent]
      Uses GPT-4o-mini to classify the query into one of four intents:
      SIMPLE_LOOKUP | POLICY_QUESTION | MULTI_HOP | COMPARISON
      │
      ▼
  [3. retrieve]
      Routes internally by intent to the right retrieval strategy:
      ┌─ SIMPLE_LOOKUP   → Graph entity lookup → Hybrid retrieval
      ├─ POLICY_QUESTION → Hybrid retrieval only
      ├─ MULTI_HOP       → Graph → Policy → Prior-Auth (3 sequential steps)
      └─ COMPARISON      → Hybrid retrieval per plan tier (Bronze / Silver / Gold)
      │
      ▼
  [4. synthesize]
      Uses GPT-4o to generate a cited, safety-compliant final answer
      │
      ▼
  [5. memory_add]     ← NEW: Persist Q&A facts to Mem0 for future sessions
      │
      ▼
    END
"""

import os
import sys
from typing import TypedDict, List
import concurrent.futures
import contextvars

# Ensure project root is on sys.path so `config` and sibling packages resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load .env FIRST — before any LangSmith/LangChain imports ─────────────────
# LANGCHAIN_TRACING_V2 must be in os.environ before langsmith is imported,
# otherwise the SDK and our _TRACING_ENABLED flag will always read False.
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from rich.console import Console

# LangSmith observability — graceful no-ops if LANGCHAIN_TRACING_V2 is not set
try:
    from langsmith import traceable as _langsmith_traceable
    _LANGSMITH_AVAILABLE = True
except ImportError:
    # If langsmith is not installed, create a passthrough decorator
    def _langsmith_traceable(*args, **kwargs):  # type: ignore[misc]
        def _decorator(fn):
            return fn
        return _decorator
    _LANGSMITH_AVAILABLE = False

from orchestration.langsmith_tracing import tag_current_run, get_run_id

from config import (
    LLM_MODEL,
    CLASSIFIER_LLM_MODEL,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
)
from orchestration.tools import policy_search, relational_search, plan_comparison_search, prior_auth_search
from orchestration.tracing import trace_log
from orchestration.memory import search_memories, add_memory

console = Console()


# ══════════════════════════════════════════════════════════════
# 1.  STATE  — the shared data bag that flows through all nodes
# ══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    What gets passed between every node in the graph.
    Each node receives this dict and returns an updated copy.
    """
    query:             str          # Original user question
    user_id:           str          # Session identifier (for logging)
    intent:            str          # Classified intent (set by classify_intent)
    retrieved_context: str          # All retrieved text (set by retrieve)
    past_memories:     str          # Relevant session memories (set by memory_search)
    memories_used:     List[str]    # Facts stored to Mem0 (set by memory_add)
    answer:            str          # Final answer (set by synthesize)
    chat_history:      List[tuple]  # [(role, message), ...] — last 5 turns
    steps_log:         List[str]    # Human-readable trace of what happened


# ══════════════════════════════════════════════════════════════
# 2.  LLM HELPERS
# ══════════════════════════════════════════════════════════════

def _classifier_llm() -> ChatOpenAI:
    """Fast, cheap model for intent classification."""
    return ChatOpenAI(
        model=CLASSIFIER_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def _synthesis_llm() -> ChatOpenAI:
    """High-accuracy model for final answer synthesis."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


# ══════════════════════════════════════════════════════════════
# 3.  NODE 1 — memory_search  (NEW)
# ══════════════════════════════════════════════════════════════

def memory_search(state: AgentState) -> AgentState:
    """NODE 1: Retrieve relevant past memories from Mem0 before classification."""
    query   = state["query"]
    user_id = state.get("user_id", "default")
    log     = list(state.get("steps_log", []))

    memories = search_memories(user_id=user_id, query=query)
    if memories:
        log.append(f"🧠 Mem0 recalled {memories.count(chr(10) + '  ')+1} memory fact(s) for user")
    else:
        log.append("🧠 Mem0: no relevant past memories found")

    return {**state, "past_memories": memories, "steps_log": log}



# ══════════════════════════════════════════════════════════════
# 4.  NODE 2 — classify_intent
# ══════════════════════════════════════════════════════════════

_INTENT_SYSTEM = """You are a query intent classifier for a Health Insurance AI assistant.
Classify the user query into EXACTLY ONE of these four categories:

  SIMPLE_LOOKUP    — Quick single-fact lookups.
                     Examples: "What is the copay for Metformin?",
                               "Is Lipitor covered?",
                               "Which cardiologists are in-network?"

  POLICY_QUESTION  — General policy, procedure, or guideline questions.
                     Examples: "How do I file a claim?",
                               "What preventive care is covered?",
                               "Do I need a referral for a specialist?"

  MULTI_HOP        — Questions needing multiple retrieval steps.
                     Examples: "Is Metformin covered and do I need prior authorization?",
                               "What is the copay for diabetes drugs and which doctors treat it?"

  COMPARISON       — Comparing plans or options.
                     Examples: "Compare Bronze and Gold deductibles",
                               "Which plan is best for a diabetic?",
                               "What are the differences between Silver and Gold?"

Respond with ONLY the category name — no explanation, no punctuation, just the word."""


def classify_intent(state: AgentState) -> AgentState:
    """NODE 1: Intent Classification."""
    llm   = _classifier_llm()
    query = state["query"]
    log   = list(state.get("steps_log", []))

    response = llm.invoke([
        SystemMessage(content=_INTENT_SYSTEM),
        HumanMessage(content=query),
    ])

    raw    = response.content.strip().upper()
    valid  = {"SIMPLE_LOOKUP", "POLICY_QUESTION", "MULTI_HOP", "COMPARISON"}
    intent = raw if raw in valid else "POLICY_QUESTION"

    log.append(f"🔍 Intent classified → {intent}")
    return {**state, "intent": intent, "steps_log": log}


# ══════════════════════════════════════════════════════════════
# 4.  NODE 2 — retrieve
# ══════════════════════════════════════════════════════════════

def retrieve(state: AgentState) -> AgentState:
    """NODE 2: Smart Retrieval."""
    query  = state["query"]
    intent = state["intent"]
    log    = list(state.get("steps_log", []))
    parts: list[str] = []

    # Initialize context-aware trace log for this node execution
    internal_logs = []
    token = trace_log.set(internal_logs)

    def _run_tool(tool_func, kwargs):
        return contextvars.copy_context().run(tool_func, kwargs)

    # ── SIMPLE_LOOKUP ────────────────────────────────────────────
    if intent == "SIMPLE_LOOKUP":
        log.append("📊 [SIMPLE_LOOKUP] Executing Graph & Hybrid retrieval concurrently")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_g = executor.submit(_run_tool, relational_search.invoke, {"query": query})
            fut_p = executor.submit(_run_tool, policy_search.invoke, {"query": query})
            
            graph_ctx = fut_g.result()
            if graph_ctx and "No structured" not in graph_ctx:
                parts.append(f"[STRUCTURED GRAPH FACTS]\n{graph_ctx}")
                log.append("🕸️ Graph entity lookup completed")
            else:
                log.append("🕸️ Graph entity lookup — no structured results")
                
            policy_ctx = fut_p.result()
            if policy_ctx and "No relevant" not in policy_ctx:
                parts.append(f"[POLICY DOCUMENTS]\n{policy_ctx}")
                log.append("📄 Hybrid policy retrieval completed")
            else:
                log.append("📄 Hybrid policy retrieval — no relevant results")

    # ── POLICY_QUESTION ──────────────────────────────────────────
    elif intent == "POLICY_QUESTION":
        log.append("📄 [POLICY_QUESTION] Full hybrid document retrieval")
        policy_ctx = policy_search.invoke({"query": query})
        if policy_ctx and "No relevant" not in policy_ctx:
            parts.append(f"[POLICY DOCUMENTS]\n{policy_ctx}")

    # ── MULTI_HOP ────────────────────────────────────────────────
    elif intent == "MULTI_HOP":
        log.append("🔗 [MULTI_HOP] Executing Graph, Policy, and Prior-Auth retrievals concurrently")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            fut_g = executor.submit(_run_tool, relational_search.invoke, {"query": query})
            fut_p = executor.submit(_run_tool, policy_search.invoke, {"query": query})
            fut_a = executor.submit(_run_tool, prior_auth_search.invoke, {"query": query})
            
            graph_ctx = fut_g.result()
            if graph_ctx and "No structured" not in graph_ctx:
                parts.append(f"[STRUCTURED GRAPH FACTS]\n{graph_ctx}")
                log.append("🕸️ Graph entity lookup completed")
            else:
                log.append("🕸️ Graph entity lookup — no structured results")
                
            policy_ctx = fut_p.result()
            if policy_ctx and "No relevant" not in policy_ctx:
                parts.append(f"[POLICY DOCUMENTS]\n{policy_ctx}")
                log.append("📄 Hybrid policy retrieval completed")
            else:
                log.append("📄 Hybrid policy retrieval — no relevant results")
                
            auth_ctx = fut_a.result()
            if auth_ctx and "No prior authorization" not in auth_ctx:
                parts.append(f"[PRIOR AUTHORIZATION RULES]\n{auth_ctx}")
                log.append("📋 Prior authorization rules retrieved")
            else:
                log.append("📋 Prior authorization — no relevant rules found")

    # ── COMPARISON ───────────────────────────────────────────────
    elif intent == "COMPARISON":
        log.append(f"⚖️  [COMPARISON] Retrieving Bronze, Silver, and Gold plan contexts concurrently")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                tier: executor.submit(_run_tool, plan_comparison_search.invoke, {"query": query, "tier": tier})
                for tier in ("Bronze", "Silver", "Gold")
            }
            for tier in ("Bronze", "Silver", "Gold"):
                tier_ctx = futures[tier].result()
                if tier_ctx and f"No {tier}" not in tier_ctx:
                    parts.append(f"[{tier.upper()} PLAN]\n{tier_ctx}")
                    log.append(f"🥇 [{tier} Tier] Retrieved plan context successfully")
                else:
                    log.append(f"⚠️ [{tier} Tier] No relevant context found")

    separator   = "\n\n" + "─" * 60 + "\n\n"
    full_context = separator.join(parts) if parts else "No relevant context found."
    
    # Capture any internal logs (like Multi-Query variants)
    for l in internal_logs:
        log.append(l)
    
    log.append(f"✅ Retrieved {len(parts)} context section(s)")
    
    # Cleanup context
    trace_log.reset(token)

    return {**state, "retrieved_context": full_context, "steps_log": log}


# ══════════════════════════════════════════════════════════════
# 6.  NODE 4 — synthesize
# ══════════════════════════════════════════════════════════════

_SYNTHESIS_TEMPLATE = """{system_prompt}

─── PAST USER MEMORIES (from previous sessions) ──────────────
{past_memories}
──────────────────────────────────────────────────────────────

─── RETRIEVED CONTEXT ────────────────────────────────────────
{context}
──────────────────────────────────────────────────────────────

─── CONVERSATION HISTORY ─────────────────────────────────────
{history}
──────────────────────────────────────────────────────────────

Using ONLY the retrieved context above, answer the user's question.
IMPORTANT: You MUST also follow any user preferences (e.g., formatting, language, or specific plan details) found in the PAST USER MEMORIES section.
If the user asks for entities with multiple criteria (e.g., "Specialist X in City Y"), you MUST verify that the same entity satisfies ALL criteria in the context. Do NOT assume an entity has a property just because it is listed in the context alongside another entity.
Always cite the source file and page number for every fact you state EXACTLY as it appears in the context (e.g., "(Source: filename.pdf, Page: 2)"). Failure to include the page number is a strict violation.
If the context does not contain enough information, say so explicitly — do NOT guess."""


def synthesize(state: AgentState) -> AgentState:
    """NODE 3: Answer Synthesis."""
    llm   = _synthesis_llm()
    query = state["query"]
    log   = list(state.get("steps_log", []))

    history_str = "\n".join(
        f"{role.upper()}: {msg}"
        for role, msg in (state.get("chat_history") or [])
    ) or "None"

    past_memories = state.get("past_memories") or "None"

    system_content = _SYNTHESIS_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        past_memories=past_memories,
        context=state["retrieved_context"],
        history=history_str,
    )

    response = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=query),
    ])

    log.append("💬 Answer synthesized")
    return {**state, "answer": response.content, "steps_log": log}


# ══════════════════════════════════════════════════════════════
# 7.  PUBLIC ORCHESTRATOR CLASS
# ══════════════════════════════════════════════════════════════

class Orchestrator:
    def __init__(self, user_id: str = "default", mem=None):
        """
        Args:
            user_id: The session ID — used for logging/tracing.
            mem:     A pre-created Mem0 Memory instance (in-memory, per-session).
                     Created by SessionManager alongside this Orchestrator.
                     Pass None to disable memory features.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment.")
        self._mem = mem                      # Mem0 in-memory instance for this session
        self.user_id: str = user_id
        self.chat_history: List[tuple] = []
        self.last_detailed_result: dict = {}

        # Bind memory helpers to this session's Memory instance so nodes can call them
        self._search_memories = lambda query: search_memories(self._mem, query)
        self._add_memory       = lambda q, a:  add_memory(self._mem, q, a)

        self.graph: any = self._build_graph()

    # ── Build graph with closures over self._mem ──────────────────────────────

    def _build_graph(self):
        """Build the LangGraph StateGraph with memory nodes bound to this session."""
        _search = self._search_memories
        _add    = self._add_memory

        def memory_search_node(state: AgentState) -> AgentState:
            """NODE 1: Retrieve relevant facts from this session's Mem0."""
            query = state["query"]
            log   = list(state.get("steps_log", []))
            memories = _search(query)
            if memories:
                count = memories.count("\n  ") + 1
                log.append(f"🧠 Mem0 recalled {count} fact(s) from this session")
            else:
                log.append("🧠 Mem0: no relevant facts recalled yet")
            return {**state, "past_memories": memories, "steps_log": log}

        def memory_add_node(state: AgentState) -> AgentState:
            """NODE 5: Persist Q&A facts back to this session's Mem0."""
            stored = _add(state["query"], state["answer"])
            log    = list(state.get("steps_log", []))
            log.append(f"💾 Mem0 stored {len(stored)} fact(s) from this turn" if stored else "💾 Mem0: no new facts extracted")
            return {**state, "memories_used": stored, "steps_log": log}

        builder = StateGraph(AgentState)
        builder.add_node("memory_search",   memory_search_node)
        builder.add_node("classify_intent", classify_intent)
        builder.add_node("retrieve",        retrieve)
        builder.add_node("synthesize",      synthesize)
        builder.add_node("memory_add",      memory_add_node)

        builder.set_entry_point("memory_search")
        builder.add_edge("memory_search",   "classify_intent")
        builder.add_edge("classify_intent", "retrieve")
        builder.add_edge("retrieve",        "synthesize")
        builder.add_edge("synthesize",      "memory_add")
        builder.add_edge("memory_add",      END)

        return builder.compile()

    def _base_state(self, query: str) -> AgentState:
        """Build the initial AgentState for a new query."""
        return {
            "query":             query,
            "user_id":           self.user_id,
            "intent":            "",
            "retrieved_context": "",
            "past_memories":     "",
            "memories_used":     [],
            "answer":            "",
            "chat_history":      self.chat_history.copy(),
            "steps_log":         [],
        }

    def ask(self, query: str, verbose: bool = False) -> str:
        result = self.graph.invoke(self._base_state(query))

        if verbose:
            console.print("\n[bold dim]📋 Orchestrator trace:[/bold dim]")
            for step in result["steps_log"]:
                console.print(f"  [dim]{step}[/dim]")

        answer = result["answer"]

        self.chat_history.append(("human", query))
        self.chat_history.append(("ai",    answer))
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        return answer

    @_langsmith_traceable(name="health-insurance-rag-query", run_type="chain")  # type: ignore[misc]
    def ask_detailed(self, query: str) -> dict:
        """
        Like ask(), but returns the full result dict for the API layer.
        Decorated with @traceable so every call creates a named top-level
        LangSmith parent span that groups all 6 LangGraph node child spans.

        Returns:
            {
                "answer":            str,
                "intent":            str,
                "steps_log":         list[str],
                "retrieved_context": str,
                "memories_used":     list[str],
                "run_id":            str | None,  # LangSmith trace UUID
            }
        """
        result = self.graph.invoke(self._base_state(query))
        answer = result["answer"]

        # ── Tag the LangSmith trace with structured metadata ──────────────────
        # Called after invoke so intent & language are already resolved.
        tag_current_run(
            session_id=self.user_id,
            intent=result.get("intent", ""),
            extra_metadata={"query_length": len(query)},
        )
        # Capture run_id while still inside the @traceable scope
        run_id = get_run_id()

        self.chat_history.append(("human", query))
        self.chat_history.append(("ai",    answer))
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        self.last_detailed_result = {
            "query":             query,
            "answer":            answer,
            "intent":            result.get("intent", "POLICY_QUESTION"),
            "steps_log":         result.get("steps_log", []),
            "retrieved_context": result.get("retrieved_context", ""),
            "memories_used":     result.get("memories_used", []),
            "run_id":            run_id,   # None when tracing is disabled
        }
        return self.last_detailed_result

    def stream_detailed(self, query: str):
        """
        Generator that yields intermediate AgentState updates as they happen.
        Useful for "Live" Developer Console views.
        """
        # Use LangGraph's streaming mode
        for event in self.graph.stream(self._base_state(query)):
            # event is a dict like {"node_name": state_update}
            for node, state in event.items():
                # We yield the current state and which node just finished
                yield {
                    "node": node,
                    "state": state
                }
        
        # After completion, update internal history (last yield will have the full state)
        # Note: In a production stream, you might want to handle this differently
        # but for this POC, the last event from 'synthesize' has the answer.


if __name__ == "__main__":
    from rich.panel import Panel
    from rich.markdown import Markdown

    orch = Orchestrator()
    q = "What is the copay for Metformin on the Silver plan?"
    console.print(f"[bold]Q:[/bold] {q}\n")
    answer = orch.ask(q, verbose=True)
    console.print(Panel(Markdown(answer), border_style="green"))
