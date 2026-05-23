"""
Health Insurance AI Copilot — LangGraph Sequential Chain Orchestrator.

Architecture (9 nodes in a directed StateGraph):

    START
      │
      ▼
  [1. query_guard]     ← NEW: PII detection, off-topic guard, query sanitization
      │ (blocked → jump to memory_add with pre-set answer)
      ▼
  [2. memory_search]   ← Retrieve relevant past facts from Mem0
      │
      ▼
  [3. classify_intent]
      Uses GPT-4o-mini to classify the query into one of four intents:
      SIMPLE_LOOKUP | POLICY_QUESTION | MULTI_HOP | COMPARISON
      │
      ▼
  [4. query_decomposer] ← NEW: Splits MULTI_HOP into atomic sub-questions
      │
      ▼
  [5. retrieve]
      Routes internally by intent to the right retrieval strategy.
      If sub_questions exist (MULTI_HOP), retrieves per sub-question.
      │
      ▼
  [6. context_quality_check] ← NEW: Validates context richness before synthesis
      │ (skip_synthesis → jump to confidence_scorer with fallback answer)
      ▼
  [7. synthesize]
      Uses GPT-4o to generate a cited, safety-compliant final answer
      │
      ▼
  [8. self_critique]   ← NEW: Verifies answer quality (Self-RAG style). Max 1 retry.
      │ (needs_retry → loop back to retrieve)
      ▼
  [9. confidence_scorer] ← NEW: Heuristic HIGH/MEDIUM/LOW confidence rating
      │
      ▼
  [10. memory_add]     ← Persist Q&A facts to Mem0 for future sessions
      │
      ▼
    END
"""

import os
import re
import sys
from typing import TypedDict, List, Optional
import concurrent.futures
import contextvars

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from rich.console import Console

try:
    from langsmith import traceable as _langsmith_traceable
    _LANGSMITH_AVAILABLE = True
except ImportError:
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
# 1.  STATE — the shared data bag that flows through all nodes
# ══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """What gets passed between every node in the graph."""
    query:              str           # Original user question
    user_id:            str           # Session identifier
    intent:             str           # Classified intent
    retrieved_context:  str           # All retrieved text
    past_memories:      str           # Relevant session memories
    memories_used:      List[str]     # Facts stored to Mem0
    answer:             str           # Final answer
    chat_history:       List[tuple]   # [(role, message), ...]
    steps_log:          List[str]     # Human-readable trace
    # ── NEW fields ────────────────────────────────────────────
    blocked:            bool          # query_guard: was query blocked?
    block_reason:       str           # query_guard: reason for blocking
    sub_questions:      List[str]     # query_decomposer: MULTI_HOP sub-questions
    skip_synthesis:     bool          # context_quality_check: skip synthesis?
    needs_retry:        bool          # self_critique: trigger retrieve retry?
    retry_count:        int           # self_critique: how many retries done
    confidence:         str           # confidence_scorer: HIGH / MEDIUM / LOW
    confidence_reason:  str           # confidence_scorer: explanation


# ══════════════════════════════════════════════════════════════
# 2.  LLM HELPERS
# ══════════════════════════════════════════════════════════════

def _classifier_llm() -> ChatOpenAI:
    """Fast, cheap model for intent classification and auxiliary LLM calls."""
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
# 3.  NODE FUNCTIONS (stateless — defined at module level)
# ══════════════════════════════════════════════════════════════

# ── PII patterns to detect and redact ─────────────────────────
_PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b',                                'SSN pattern'),
    (r'\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b', 'date-of-birth pattern'),
    (r'\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',   'credit/debit card pattern'),
    (r'\bMRN[\s:#]*\d{6,}\b',                                 'medical record number'),
]

# ── Off-topic topic signal words ───────────────────────────────
_OFF_TOPIC_SIGNALS = [
    'weather', 'forecast', 'cryptocurrency', 'bitcoin', 'stock price',
    'sports score', 'movie review', 'recipe', 'video game', 'dating',
    'political party', 'election result',
]

def classify_intent(state: AgentState) -> AgentState:
    """NODE 3: Intent Classification."""
    llm   = _classifier_llm()
    query = state["query"]
    log   = list(state.get("steps_log", []))

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

    response = llm.invoke([
        SystemMessage(content=_INTENT_SYSTEM),
        HumanMessage(content=query),
    ])

    raw    = response.content.strip().upper()
    valid  = {"SIMPLE_LOOKUP", "POLICY_QUESTION", "MULTI_HOP", "COMPARISON"}
    intent = raw if raw in valid else "POLICY_QUESTION"

    log.append(f"🔍 Intent classified → {intent}")
    return {**state, "intent": intent, "steps_log": log}


def retrieve(state: AgentState) -> AgentState:
    """NODE 5: Smart Retrieval — routes by intent, uses sub-questions when available."""
    query         = state["query"]
    intent        = state["intent"]
    sub_questions = state.get("sub_questions", [])
    retry_count   = state.get("retry_count", 0)
    log           = list(state.get("steps_log", []))
    parts: list[str] = []

    internal_logs = []
    token = trace_log.set(internal_logs)

    def _run_tool(tool_func, kwargs):
        return contextvars.copy_context().run(tool_func, kwargs)

    if retry_count > 0:
        log.append(f"🔄 Retry #{retry_count}: re-running retrieval with broader search")

    # ── SIMPLE_LOOKUP ──────────────────────────────────────────
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

    # ── POLICY_QUESTION ────────────────────────────────────────
    elif intent == "POLICY_QUESTION":
        log.append("📄 [POLICY_QUESTION] Full hybrid document retrieval")
        policy_ctx = policy_search.invoke({"query": query})
        if policy_ctx and "No relevant" not in policy_ctx:
            parts.append(f"[POLICY DOCUMENTS]\n{policy_ctx}")

    # ── MULTI_HOP ──────────────────────────────────────────────
    elif intent == "MULTI_HOP":
        # Use decomposed sub-questions for targeted retrieval per hop
        queries_to_run = sub_questions if sub_questions else [query]
        log.append(f"🔗 [MULTI_HOP] Executing {len(queries_to_run)} targeted retrieval(s)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Always run graph, policy, and prior-auth in parallel on main query
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

        # Additionally run per-sub-question retrieval for targeted coverage
        if len(sub_questions) > 1:
            log.append(f"✂️ Running targeted retrieval for {len(sub_questions)} sub-question(s)")
            for i, sq in enumerate(sub_questions):
                sq_ctx = policy_search.invoke({"query": sq})
                if sq_ctx and "No relevant" not in sq_ctx:
                    parts.append(f"[SUB-QUESTION {i+1}: {sq}]\n{sq_ctx}")
                    log.append(f"  ↳ Sub-Q {i+1} retrieved successfully")

    # ── COMPARISON ─────────────────────────────────────────────
    elif intent == "COMPARISON":
        log.append("⚖️  [COMPARISON] Retrieving Bronze, Silver, and Gold plan contexts concurrently")
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

    separator    = "\n\n" + "─" * 60 + "\n\n"
    full_context = separator.join(parts) if parts else "No relevant context found."

    for l in internal_logs:
        log.append(l)

    log.append(f"✅ Retrieved {len(parts)} context section(s)")
    trace_log.reset(token)

    return {**state, "retrieved_context": full_context, "steps_log": log}


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
IMPORTANT: You MUST also follow any user preferences found in the PAST USER MEMORIES section.
If the user asks for entities with multiple criteria, verify ALL criteria match in the context.
Always cite the source file and page number for every fact EXACTLY as it appears (e.g., "(Source: filename.pdf, Page: 2)").
If the context does not contain enough information, say so explicitly — do NOT guess."""


def synthesize(state: AgentState) -> AgentState:
    """NODE 7: Answer Synthesis."""
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
# 4.  ORCHESTRATOR CLASS — builds the LangGraph with all 9 nodes
# ══════════════════════════════════════════════════════════════

class Orchestrator:
    def __init__(self, user_id: str = "default", mem=None):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment.")
        self._mem = mem
        self.user_id: str = user_id
        self.chat_history: List[tuple] = []
        self.last_detailed_result: dict = {}

        self._search_memories = lambda query: search_memories(self._mem, query)
        self._add_memory       = lambda q, a:  add_memory(self._mem, q, a)

        self.graph: any = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph with 9 nodes, conditional edges, and retry loop."""
        _search = self._search_memories
        _add    = self._add_memory

        # ── Node 1: Query Guard ─────────────────────────────────
        def query_guard_node(state: AgentState) -> AgentState:
            """PII detection, off-topic guard, query sanitization."""
            query = state["query"]
            log   = list(state.get("steps_log", []))

            # 1a. PII detection — redact but don't block (still useful info to answer)
            sanitized = query
            pii_found = []
            for pattern, label in _PII_PATTERNS:
                if re.search(pattern, query, re.IGNORECASE):
                    sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
                    pii_found.append(label)

            if pii_found:
                log.append(f"🛡️ Guard: PII detected ({', '.join(pii_found)}) — sensitive data redacted")
                # Don't block, just sanitize
                return {**state, "query": sanitized, "blocked": False, "block_reason": "",
                        "sub_questions": [], "skip_synthesis": False, "needs_retry": False,
                        "retry_count": 0, "confidence": "", "confidence_reason": "",
                        "steps_log": log}

            # 1b. Off-topic detection
            q_lower = query.lower()
            for signal in _OFF_TOPIC_SIGNALS:
                if signal in q_lower:
                    log.append(f"🛡️ Guard BLOCKED: off-topic signal '{signal}' detected")
                    return {
                        **state,
                        "blocked":     True,
                        "block_reason": "OFF_TOPIC",
                        "answer":      (
                            f"I'm a Health Insurance AI assistant and can only help with health "
                            f"insurance questions (coverage, claims, providers, plan comparisons, "
                            f"drug formularies, etc.). Your question appears to be about '{signal}' "
                            f"which is outside my scope. Please ask something related to your health plan."
                        ),
                        "sub_questions":   [],
                        "skip_synthesis":  True,
                        "needs_retry":     False,
                        "retry_count":     0,
                        "confidence":      "BLOCKED",
                        "confidence_reason": "Off-topic query",
                        "steps_log": log,
                    }

            # 1c. Empty / trivially short query
            if len(query.strip()) < 4:
                log.append("🛡️ Guard BLOCKED: query too short or empty")
                return {
                    **state,
                    "blocked":     True,
                    "block_reason": "INVALID_QUERY",
                    "answer":      "Please provide a more detailed health insurance question.",
                    "sub_questions":   [],
                    "skip_synthesis":  True,
                    "needs_retry":     False,
                    "retry_count":     0,
                    "confidence":      "BLOCKED",
                    "confidence_reason": "Query too short",
                    "steps_log": log,
                }

            log.append("✅ Guard: query validated — all checks passed")
            return {
                **state,
                "blocked":     False,
                "block_reason": "",
                "sub_questions":   [],
                "skip_synthesis":  False,
                "needs_retry":     False,
                "retry_count":     state.get("retry_count", 0),
                "confidence":      "",
                "confidence_reason": "",
                "steps_log": log,
            }

        # ── Node 2: Memory Search ───────────────────────────────
        def memory_search_node(state: AgentState) -> AgentState:
            """Retrieve relevant facts from this session's Mem0."""
            query    = state["query"]
            log      = list(state.get("steps_log", []))
            memories = _search(query)
            if memories:
                count = memories.count("\n  ") + 1
                log.append(f"🧠 Mem0 recalled {count} fact(s) from this session")
            else:
                log.append("🧠 Mem0: no relevant facts recalled yet")
            return {**state, "past_memories": memories, "steps_log": log}

        # ── Node 4: Query Decomposer ────────────────────────────
        def query_decomposer_node(state: AgentState) -> AgentState:
            """Split MULTI_HOP queries into atomic sub-questions for targeted retrieval."""
            intent = state.get("intent", "")
            log    = list(state.get("steps_log", []))

            # Only decompose MULTI_HOP — pass-through for everything else
            if intent != "MULTI_HOP":
                log.append(f"✂️ Decomposer: skipped (intent={intent}, no decomposition needed)")
                return {**state, "sub_questions": [], "steps_log": log}

            query = state["query"]
            llm   = _classifier_llm()

            prompt = (
                f"Break this health insurance question into 2-4 atomic sub-questions "
                f"that can each be answered independently from separate document lookups.\n\n"
                f"Question: {query}\n\n"
                f"Rules:\n"
                f"- Each sub-question must be self-contained (no pronouns referencing other sub-questions)\n"
                f"- Focus on ONE entity or fact per sub-question\n"
                f"- Return ONLY the sub-questions, one per line, no numbering or bullet points"
            )

            try:
                response = llm.invoke(prompt)
                sub_questions = [q.strip() for q in response.content.splitlines() if q.strip()]
                # Fallback: if decomposition fails or returns nothing useful
                if not sub_questions or len(sub_questions) == 1:
                    sub_questions = [query]
                    log.append(f"✂️ Decomposer: could not split — using original query")
                else:
                    log.append(f"✂️ Decomposer: split into {len(sub_questions)} sub-question(s)")
                    for i, sq in enumerate(sub_questions):
                        log.append(f"  ↳ Sub-Q {i+1}: {sq}")
            except Exception as e:
                sub_questions = [query]
                log.append(f"✂️ Decomposer: error ({e}) — using original query")

            return {**state, "sub_questions": sub_questions, "steps_log": log}

        # ── Node 6: Context Quality Check ───────────────────────
        def context_quality_check_node(state: AgentState) -> AgentState:
            """Validate retrieved context richness before synthesis."""
            log = list(state.get("steps_log", []))
            ctx = state.get("retrieved_context", "")

            # Count meaningful sections
            empty_phrases = [
                "No relevant context found.",
                "No structured relational data found",
                "No relevant policy information found",
            ]
            is_empty = (
                not ctx or
                ctx.strip() in empty_phrases or
                all(ep in ctx for ep in ["No relevant"]) or
                len(ctx.strip()) < 80
            )

            if is_empty:
                log.append("⚠️ Context Quality: insufficient context retrieved — synthesis skipped")
                return {
                    **state,
                    "skip_synthesis": True,
                    "answer": (
                        "I don't have sufficient information in my knowledge base to answer this "
                        "question accurately. This could be because:\n"
                        "• The specific plan detail isn't in our documents\n"
                        "• The question may need more specific terms\n\n"
                        "Please contact your insurance provider directly or refer to your plan documents."
                    ),
                    "confidence":        "LOW",
                    "confidence_reason": "No relevant context retrieved",
                    "steps_log": log,
                }

            # Count context sections for logging
            section_count = ctx.count("─" * 20) + 1
            char_count    = len(ctx)
            log.append(
                f"✅ Context Quality: {section_count} section(s) | {char_count:,} chars — synthesis proceeding"
            )
            return {**state, "skip_synthesis": False, "steps_log": log}

        # ── Node 8: Self-Critique ───────────────────────────────
        def self_critique_node(state: AgentState) -> AgentState:
            """Self-RAG style reflection: verify answer quality, trigger retry if poor."""
            log         = list(state.get("steps_log", []))
            retry_count = state.get("retry_count", 0)

            # Skip if already retried, blocked, or synthesis was skipped
            if retry_count >= 1 or state.get("skip_synthesis"):
                log.append("🔄 Self-Critique: skipped (max retries reached or synthesis was skipped)")
                return {**state, "needs_retry": False, "steps_log": log}

            answer = state.get("answer", "")
            query  = state["query"]

            if not answer:
                log.append("🔄 Self-Critique: no answer to evaluate")
                return {**state, "needs_retry": False, "steps_log": log}

            llm = _classifier_llm()
            prompt = (
                f"You are evaluating a health insurance AI assistant's answer quality.\n\n"
                f"Question: {query}\n\n"
                f"Answer (first 600 chars): {answer[:600]}\n\n"
                f"Evaluate:\n"
                f"1. Does the answer directly address the specific question?\n"
                f"2. Does it contain concrete details (numbers, plan names, coverage specifics)?\n"
                f"3. Does it cite sources?\n\n"
                f"Respond with ONLY one word:\n"
                f"- GOOD: answer is adequate, specific, and addresses the question\n"
                f"- RETRY: answer is vague, generic, or says 'I don't have info' when a better retrieval might help"
            )

            try:
                response = llm.invoke(prompt)
                verdict  = response.content.strip().upper()

                if "RETRY" in verdict:
                    log.append(f"🔄 Self-Critique: answer quality INSUFFICIENT — triggering retry #{retry_count + 1}")
                    return {**state, "needs_retry": True, "retry_count": retry_count + 1, "steps_log": log}
                else:
                    log.append("✅ Self-Critique: answer quality VERIFIED — proceeding")
                    return {**state, "needs_retry": False, "steps_log": log}
            except Exception as e:
                log.append(f"🔄 Self-Critique: evaluation error ({e}) — skipping retry")
                return {**state, "needs_retry": False, "steps_log": log}

        # ── Node 9: Confidence Scorer ───────────────────────────
        def confidence_scorer_node(state: AgentState) -> AgentState:
            """Heuristic HIGH/MEDIUM/LOW confidence rating based on context and answer quality."""
            log = list(state.get("steps_log", []))

            # Already set during guard or quality check
            if state.get("confidence") in ("BLOCKED", "LOW"):
                log.append(f"📊 Confidence: {state['confidence']} (pre-set by earlier node)")
                return {**state, "steps_log": log}

            answer  = state.get("answer", "")
            ctx     = state.get("retrieved_context", "")
            reasons = []
            score   = 0

            # Citations in answer
            citation_count = answer.count("(Source:")
            if citation_count >= 3:
                score += 3;  reasons.append(f"{citation_count} citations")
            elif citation_count >= 1:
                score += 2;  reasons.append(f"{citation_count} citation(s)")
            else:
                score -= 1;  reasons.append("no citations found")

            # Context richness
            ctx_len = len(ctx)
            if ctx_len > 2000:
                score += 2;  reasons.append("rich context")
            elif ctx_len > 500:
                score += 1;  reasons.append("moderate context")
            else:
                reasons.append("sparse context")

            # Hedging language in answer (uncertainty signals)
            hedging_phrases = [
                "i don't have", "cannot find", "not available in",
                "no information", "unable to", "not in my knowledge"
            ]
            if any(h in answer.lower() for h in hedging_phrases):
                score -= 1;  reasons.append("hedged answer")

            # Retry needed (lower confidence if answer required retry)
            if state.get("retry_count", 0) > 0:
                score -= 1;  reasons.append("required retry")

            # Final rating
            if score >= 4:
                confidence = "HIGH"
            elif score >= 2:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            reason_str = " | ".join(reasons)
            log.append(f"📊 Confidence: {confidence} (score={score} — {reason_str})")

            return {**state, "confidence": confidence, "confidence_reason": reason_str, "steps_log": log}

        # ── Node 10: Memory Add ─────────────────────────────────
        def memory_add_node(state: AgentState) -> AgentState:
            """Persist Q&A facts back to this session's Mem0."""
            # Skip memory storage if query was blocked
            if state.get("blocked"):
                log = list(state.get("steps_log", []))
                log.append("💾 Mem0: skipped (query was blocked)")
                return {**state, "memories_used": [], "steps_log": log}

            stored = _add(state["query"], state["answer"])
            log    = list(state.get("steps_log", []))
            if stored:
                log.append(f"💾 Mem0 stored {len(stored)} fact(s) from this turn")
            else:
                log.append("💾 Mem0: no new facts extracted")
            return {**state, "memories_used": stored, "steps_log": log}

        # ── Build the StateGraph ────────────────────────────────
        builder = StateGraph(AgentState)

        builder.add_node("query_guard",           query_guard_node)
        builder.add_node("memory_search",          memory_search_node)
        builder.add_node("classify_intent",        classify_intent)
        builder.add_node("query_decomposer",       query_decomposer_node)
        builder.add_node("retrieve",               retrieve)
        builder.add_node("context_quality_check",  context_quality_check_node)
        builder.add_node("synthesize",             synthesize)
        builder.add_node("self_critique",          self_critique_node)
        builder.add_node("confidence_scorer",      confidence_scorer_node)
        builder.add_node("memory_add",             memory_add_node)

        builder.set_entry_point("query_guard")

        # Conditional: blocked queries skip straight to memory_add
        builder.add_conditional_edges(
            "query_guard",
            lambda s: "blocked" if s.get("blocked") else "ok",
            {"blocked": "memory_add", "ok": "memory_search"},
        )

        builder.add_edge("memory_search",   "classify_intent")
        builder.add_edge("classify_intent", "query_decomposer")
        builder.add_edge("query_decomposer","retrieve")
        builder.add_edge("retrieve",        "context_quality_check")

        # Conditional: no context → skip synthesis, go straight to confidence_scorer
        builder.add_conditional_edges(
            "context_quality_check",
            lambda s: "skip" if s.get("skip_synthesis") else "ok",
            {"skip": "confidence_scorer", "ok": "synthesize"},
        )

        builder.add_edge("synthesize", "self_critique")

        # Conditional: poor answer → retry retrieve (max once)
        builder.add_conditional_edges(
            "self_critique",
            lambda s: "retry" if s.get("needs_retry") else "ok",
            {"retry": "retrieve", "ok": "confidence_scorer"},
        )

        builder.add_edge("confidence_scorer", "memory_add")
        builder.add_edge("memory_add",         END)

        return builder.compile()

    # ── Base state builder ────────────────────────────────────────────────────

    def _base_state(self, query: str) -> AgentState:
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
            # New fields — always initialized
            "blocked":           False,
            "block_reason":      "",
            "sub_questions":     [],
            "skip_synthesis":    False,
            "needs_retry":       False,
            "retry_count":       0,
            "confidence":        "",
            "confidence_reason": "",
        }

    # ── Public API ────────────────────────────────────────────────────────────

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
        """Like ask(), but returns the full result dict for the API layer."""
        result = self.graph.invoke(self._base_state(query))
        answer = result["answer"]

        tag_current_run(
            session_id=self.user_id,
            intent=result.get("intent", ""),
            extra_metadata={
                "query_length": len(query),
                "blocked":      result.get("blocked", False),
                "confidence":   result.get("confidence", ""),
            },
        )
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
            "run_id":            run_id,
            "blocked":           result.get("blocked", False),
            "confidence":        result.get("confidence", ""),
            "confidence_reason": result.get("confidence_reason", ""),
            "sub_questions":     result.get("sub_questions", []),
        }
        return self.last_detailed_result

    def stream_detailed(self, query: str):
        """Generator that yields intermediate AgentState updates for the Developer Console."""
        for event in self.graph.stream(self._base_state(query)):
            for node, state in event.items():
                yield {"node": node, "state": state}


if __name__ == "__main__":
    from rich.panel import Panel
    from rich.markdown import Markdown

    orch = Orchestrator()
    q = "What is the copay for Metformin on the Silver plan?"
    console.print(f"[bold]Q:[/bold] {q}\n")
    answer = orch.ask(q, verbose=True)
    console.print(Panel(Markdown(answer), border_style="green"))
