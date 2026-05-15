"""
Health Insurance AI Copilot — LangGraph Sequential Chain Orchestrator.

Architecture (3 nodes in a directed StateGraph):

    START
      │
      ▼
  [1. classify_intent]
      Uses GPT-4o-mini to classify the query into one of four intents:
      SIMPLE_LOOKUP | POLICY_QUESTION | MULTI_HOP | COMPARISON
      │
      ▼
  [2. retrieve]
      Routes internally by intent to the right retrieval strategy:
      ┌─ SIMPLE_LOOKUP   → Graph entity lookup → Hybrid retrieval
      ├─ POLICY_QUESTION → Hybrid retrieval only
      ├─ MULTI_HOP       → Graph → Policy → Prior-Auth (3 sequential steps)
      └─ COMPARISON      → Hybrid retrieval per plan tier (Bronze / Silver / Gold)
      │
      ▼
  [3. synthesize]
      Uses GPT-4o to generate a cited, safety-compliant final answer
      │
      ▼
    END
"""

import os
import sys
from typing import TypedDict, List

# Ensure project root is on sys.path so `config` and sibling packages resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from rich.console import Console

from config import (
    LLM_MODEL,
    CLASSIFIER_LLM_MODEL,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
)
from orchestration.tools import policy_search, relational_search, plan_comparison_search, prior_auth_search

load_dotenv()
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
    intent:            str          # Classified intent (set by classify_intent)
    retrieved_context: str          # All retrieved text (set by retrieve)
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
# 3.  NODE 1 — classify_intent
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

    # ── SIMPLE_LOOKUP ────────────────────────────────────────────
    if intent == "SIMPLE_LOOKUP":
        log.append("📊 [SIMPLE_LOOKUP] Step 1 — Graph entity lookup")
        graph_ctx = relational_search.invoke({"query": query})
        if graph_ctx and "No structured" not in graph_ctx:
            parts.append(f"[STRUCTURED GRAPH FACTS]\n{graph_ctx}")

        log.append("📄 [SIMPLE_LOOKUP] Step 2 — Hybrid document retrieval")
        policy_ctx = policy_search.invoke({"query": query})
        if policy_ctx and "No relevant" not in policy_ctx:
            parts.append(f"[POLICY DOCUMENTS]\n{policy_ctx}")

    # ── POLICY_QUESTION ──────────────────────────────────────────
    elif intent == "POLICY_QUESTION":
        log.append("📄 [POLICY_QUESTION] Full hybrid document retrieval")
        policy_ctx = policy_search.invoke({"query": query})
        if policy_ctx and "No relevant" not in policy_ctx:
            parts.append(f"[POLICY DOCUMENTS]\n{policy_ctx}")

    # ── MULTI_HOP ────────────────────────────────────────────────
    elif intent == "MULTI_HOP":
        log.append("🔗 [MULTI_HOP] Step 1 — Graph entity lookup")
        graph_ctx = relational_search.invoke({"query": query})
        if graph_ctx and "No structured" not in graph_ctx:
            parts.append(f"[STRUCTURED GRAPH FACTS]\n{graph_ctx}")

        log.append("🔗 [MULTI_HOP] Step 2 — Hybrid policy retrieval")
        policy_ctx = policy_search.invoke({"query": query})
        if policy_ctx and "No relevant" not in policy_ctx:
            parts.append(f"[POLICY DOCUMENTS]\n{policy_ctx}")

        log.append("🔗 [MULTI_HOP] Step 3 — Prior authorization check")
        auth_ctx = prior_auth_search.invoke({"query": query})
        if auth_ctx and "No prior authorization" not in auth_ctx:
            parts.append(f"[PRIOR AUTHORIZATION RULES]\n{auth_ctx}")

    # ── COMPARISON ───────────────────────────────────────────────
    elif intent == "COMPARISON":
        for tier in ("Bronze", "Silver", "Gold"):
            log.append(f"⚖️  [COMPARISON] Retrieving {tier} plan context")
            tier_ctx = plan_comparison_search.invoke({"query": query, "tier": tier})
            if tier_ctx and f"No {tier}" not in tier_ctx:
                parts.append(f"[{tier.upper()} PLAN]\n{tier_ctx}")

    separator   = "\n\n" + "─" * 60 + "\n\n"
    full_context = separator.join(parts) if parts else "No relevant context found."
    log.append(f"✅ Retrieved {len(parts)} context section(s)")

    return {**state, "retrieved_context": full_context, "steps_log": log}


# ══════════════════════════════════════════════════════════════
# 5.  NODE 3 — synthesize
# ══════════════════════════════════════════════════════════════

_SYNTHESIS_TEMPLATE = """{system_prompt}

─── RETRIEVED CONTEXT ────────────────────────────────────────
{context}
──────────────────────────────────────────────────────────────

─── CONVERSATION HISTORY ─────────────────────────────────────
{history}
──────────────────────────────────────────────────────────────

Using ONLY the retrieved context above, answer the user's question.
If the user asks for entities with multiple criteria (e.g., "Specialist X in City Y"), you MUST verify that the same entity satisfies ALL criteria in the context. Do NOT assume an entity has a property just because it is listed in the context alongside another entity.
Always cite the source file and page number for every fact you state.
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

    system_content = _SYNTHESIS_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
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
# 6.  BUILD THE LANGGRAPH STATEGRAPH
# ══════════════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve",        retrieve)
    builder.add_node("synthesize",      synthesize)

    builder.set_entry_point("classify_intent")
    builder.add_edge("classify_intent", "retrieve")
    builder.add_edge("retrieve",        "synthesize")
    builder.add_edge("synthesize",      END)

    return builder.compile()


# ══════════════════════════════════════════════════════════════
# 7.  PUBLIC ORCHESTRATOR CLASS
# ══════════════════════════════════════════════════════════════

class Orchestrator:
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment.")
        self.graph: any = build_graph()
        self.chat_history: List[tuple] = []

    def ask(self, query: str, verbose: bool = False) -> str:
        initial_state: AgentState = {
            "query":             query,
            "intent":            "",
            "retrieved_context": "",
            "answer":            "",
            "chat_history":      self.chat_history.copy(),
            "steps_log":         [],
        }

        result = self.graph.invoke(initial_state)

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


if __name__ == "__main__":
    from rich.panel import Panel
    from rich.markdown import Markdown

    orch = Orchestrator()
    q = "What is the copay for Metformin on the Silver plan?"
    console.print(f"[bold]Q:[/bold] {q}\n")
    answer = orch.ask(q, verbose=True)
    console.print(Panel(Markdown(answer), border_style="green"))
