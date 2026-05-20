import os
import streamlit as st
import httpx
import uuid
import asyncio
from datetime import datetime
from config import API_BASE_URL, APP_TITLE, APP_SUBTITLE, THEME, EXAMPLE_QUERIES, INTENT_METADATA

# ══════════════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# 2. GLOBAL CSS — Premium dark glassmorphism design
# ══════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --primary:    {THEME["primary"]};
        --secondary:  {THEME["secondary"]};
        --bg:         {THEME["background"]};
        --card-bg:    {THEME["card_bg"]};
        --text:       {THEME["text"]};
        --border:     rgba(255,255,255,0.08);
        --glow:       rgba(99,102,241,0.3);
        --user-grad:  linear-gradient(135deg, #4f46e5, #7c3aed);
        --ai-bg:      rgba(255,255,255,0.04);
        --success:    #10b981;
        --warning:    #f59e0b;
        --error:      #ef4444;
    }}

    /* ── Base ────────────────────────────────── */
    .stApp {{
        background: radial-gradient(ellipse at top, #0f1729 0%, #0a0f1e 60%, #050810 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 1.5rem 2rem 4rem 2rem; max-width: 1200px; }}

    /* ── Sidebar ─────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(10,15,35,0.98) 0%, rgba(5,8,20,0.99) 100%);
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] .block-container {{ padding: 1.5rem 1rem; }}

    /* ── Sidebar Brand ───────────────────────── */
    .brand-header {{
        display: flex; align-items: center; gap: 12px;
        padding: 12px 0 20px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 20px;
    }}
    .brand-logo {{
        width: 44px; height: 44px; border-radius: 12px;
        background: var(--user-grad);
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; box-shadow: 0 4px 15px var(--glow);
    }}
    .brand-title {{ font-size: 1.05rem; font-weight: 700; color: white; line-height: 1.2; }}
    .brand-sub   {{ font-size: 0.72rem; color: #64748b; }}

    /* ── Sidebar Pills ───────────────────────── */
    .sidebar-section {{ font-size: 0.7rem; font-weight: 600; color: #475569;
        text-transform: uppercase; letter-spacing: 0.1em; margin: 20px 0 10px 0; }}

    .session-pill {{
        background: rgba(255,255,255,0.04); border: 1px solid var(--border);
        border-radius: 8px; padding: 8px 12px; font-size: 0.78rem;
        color: #94a3b8; font-family: 'JetBrains Mono', monospace;
        margin-bottom: 12px; word-break: break-all; line-height: 1.4;
    }}

    /* ── Dev Console Button ──────────────────── */
    .dev-console-btn, .dev-console-btn:link, .dev-console-btn:visited {{
        display: flex; align-items: center; justify-content: center; gap: 8px;
        width: 100%; padding: 11px 14px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 10px; border: none; cursor: pointer;
        color: white !important; font-weight: 700; font-size: 0.86rem;
        text-decoration: none !important; transition: all 0.25s ease;
        box-shadow: 0 4px 18px rgba(99,102,241,0.4);
        margin-bottom: 8px; letter-spacing: 0.01em;
    }}
    .dev-console-btn:hover {{
        box-shadow: 0 6px 24px rgba(99,102,241,0.6);
        transform: translateY(-2px);
        color: white !important;
    }}

    /* ── Example Query Buttons ───────────────── */
    .stButton > button {{
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #94a3b8 !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        padding: 8px 12px !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover {{
        background: rgba(99,102,241,0.15) !important;
        border-color: rgba(99,102,241,0.4) !important;
        color: white !important;
    }}

    /* ── Main Header ─────────────────────────── */
    .main-hero {{
        text-align: center; padding: 40px 20px 32px 20px;
        background: linear-gradient(135deg, rgba(79,70,229,0.08) 0%, rgba(124,58,237,0.05) 100%);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 20px; margin-bottom: 28px;
        backdrop-filter: blur(10px);
    }}
    .hero-icon  {{ font-size: 3.5rem; display: block; margin-bottom: 12px; }}
    .hero-title {{ font-size: 2rem; font-weight: 700; color: white; margin-bottom: 8px;
        background: linear-gradient(135deg, #fff 30%, #a5b4fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .hero-sub   {{ font-size: 1rem; color: #64748b; }}

    /* ── Chat Container ──────────────────────── */
    .chat-wrapper {{
        display: flex; flex-direction: column; gap: 16px;
        padding-bottom: 12px;
    }}

    /* ── User Bubble ─────────────────────────── */
    .msg-user {{
        display: flex; justify-content: flex-end; gap: 10px; align-items: flex-start;
    }}
    .msg-user .bubble {{
        background: var(--user-grad);
        color: white; border-radius: 18px 18px 4px 18px;
        padding: 13px 18px; max-width: 72%; font-size: 0.93rem;
        line-height: 1.6; box-shadow: 0 4px 16px rgba(79,70,229,0.3);
        animation: slideUp 0.3s ease-out;
    }}
    .msg-user .avatar {{
        width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
        background: var(--user-grad); display: flex; align-items: center;
        justify-content: center; font-size: 16px; margin-top: 2px;
        box-shadow: 0 2px 8px rgba(79,70,229,0.4);
    }}

    /* ── AI Bubble ───────────────────────────── */
    .msg-ai {{
        display: flex; justify-content: flex-start; gap: 10px; align-items: flex-start;
    }}
    .msg-ai .avatar {{
        width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid rgba(255,255,255,0.12);
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; margin-top: 2px;
    }}
    .msg-ai .bubble-wrapper {{ flex: 1; max-width: 82%; animation: slideUp 0.3s ease-out; }}
    .msg-ai .bubble {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px 18px 18px 4px;
        padding: 15px 18px; font-size: 0.93rem; line-height: 1.7;
        color: #e2e8f0; backdrop-filter: blur(8px);
    }}

    /* ── Intent Badge ────────────────────────── */
    .intent-row {{
        display: flex; align-items: center; gap: 8px;
        margin-bottom: 10px; flex-wrap: wrap;
    }}
    .intent-badge {{
        font-size: 0.68rem; font-weight: 700; padding: 3px 10px;
        border-radius: 20px; text-transform: uppercase; letter-spacing: 0.08em;
        display: inline-flex; align-items: center; gap: 4px;
    }}
    .ts-badge {{
        font-size: 0.68rem; color: #475569;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* ── Pipeline Trace ──────────────────────── */
    .pipeline-trace {{
        margin-top: 12px; padding: 12px 14px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
    }}
    .pipeline-title {{
        font-size: 0.72rem; font-weight: 600; color: #475569;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;
    }}
    .pipeline-steps {{
        display: flex; flex-direction: column; gap: 4px;
    }}
    .pipeline-step {{
        font-size: 0.78rem; color: #64748b;
        font-family: 'JetBrains Mono', monospace;
        padding: 2px 0;
        display: flex; align-items: center; gap: 6px;
    }}

    /* ── LangSmith Badge ─────────────────────── */
    .langsmith-link {{
        display: inline-flex; align-items: center; gap: 5px;
        font-size: 0.7rem; color: #4f46e5; text-decoration: none;
        background: rgba(79,70,229,0.1); border: 1px solid rgba(79,70,229,0.25);
        border-radius: 6px; padding: 2px 8px;
        transition: all 0.2s ease;
    }}
    .langsmith-link:hover {{
        background: rgba(79,70,229,0.2); color: #818cf8;
    }}

    /* ── Memories Used Chip ──────────────────── */
    .memory-chip {{
        font-size: 0.7rem; background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.25); color: #10b981;
        border-radius: 6px; padding: 2px 8px; display: inline-flex;
        align-items: center; gap: 4px;
    }}

    /* ── Status strip ────────────────────────── */
    .status-strip {{
        display: flex; align-items: center; gap: 6px;
        font-size: 0.78rem; color: #10b981;
        margin-bottom: 16px; padding: 8px 14px;
        background: rgba(16,185,129,0.06);
        border: 1px solid rgba(16,185,129,0.15); border-radius: 8px;
    }}
    .status-dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: #10b981; animation: pulse 2s infinite;
    }}

    /* ── Divider ─────────────────────────────── */
    .chat-divider {{
        border: none; border-top: 1px solid rgba(255,255,255,0.05);
        margin: 8px 0;
    }}

    /* ── Animations ──────────────────────────── */
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50%       {{ opacity: 0.4; }}
    }}

    /* ── Bottom block wrapper transparency ────── */
    div[data-testid="stBottom"] {{
        background-color: transparent !important;
        background: transparent !important;
    }}
    div[data-testid="stBottom"] > div {{
        background-color: transparent !important;
        background: transparent !important;
    }}

    /* ── Input area ──────────────────────────── */
    div[data-testid="stChatInput"] > div {{
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }}
    /* Make descendant wrappers transparent to reveal glassmorphic background */
    div[data-testid="stChatInput"] > div div {{
        background-color: transparent !important;
        background: transparent !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        color: white !important;
        background: transparent !important;
        background-color: transparent !important;
        font-family: 'Inter', sans-serif !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: #64748b !important;
    }}
    div[data-testid="stChatInput"] button {{
        color: #94a3b8 !important;
        background-color: transparent !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stChatInput"] button:hover {{
        color: #818cf8 !important;
        transform: scale(1.05) !important;
    }}

    /* ── Status box ──────────────────────────── */
    div[data-testid="stStatusWidget"] {{
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }}

    /* ── Expander ────────────────────────────── */
    .streamlit-expanderHeader {{
        background: rgba(255,255,255,0.03) !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        color: #64748b !important;
    }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 3. SESSION STATE
# ══════════════════════════════════════════════════════════════

if "session_id"  not in st.session_state: st.session_state.session_id  = str(uuid.uuid4())
if "messages"    not in st.session_state: st.session_state.messages    = []
if "is_loading"  not in st.session_state: st.session_state.is_loading  = False
if "total_queries" not in st.session_state: st.session_state.total_queries = 0

# ══════════════════════════════════════════════════════════════
# 4. API HELPERS
# ══════════════════════════════════════════════════════════════

async def call_chat_api(query: str):
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            payload = {"session_id": st.session_state.session_id, "query": query}
            if "plan_tier" in st.session_state:
                payload["plan_tier"] = st.session_state.plan_tier
                
            resp = await client.post(f"{API_BASE_URL}/chat", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            st.error(f"Backend error: {e}")
            return None

async def get_health():
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{API_BASE_URL}/health")
            return r.json()
        except:
            return None

# ══════════════════════════════════════════════════════════════
# 5. SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    # Brand
    st.markdown(f"""
    <div class="brand-header">
        <div class="brand-logo">🛡️</div>
        <div>
            <div class="brand-title">{APP_TITLE}</div>
            <div class="brand-sub">{APP_SUBTITLE}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dev Console shortcut
    console_url = "/dev-console" if (
        os.getenv("RUN_MONOLITH","false").lower() == "true"
        or "localhost:8000" not in API_BASE_URL
    ) else "http://localhost:8000/dev-console"

    st.markdown(f"""
    <a href="{console_url}" target="_blank" class="dev-console-btn">
        🚀&nbsp; Open Developer Console
    </a>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="session-pill">
        <span style="color:#475569;">ID&nbsp;</span>
        {st.session_state.session_id[:8]}…{st.session_state.session_id[-4:]}
    </div>
    """, unsafe_allow_html=True)
    
    st.selectbox(
        "Plan Tier", 
        ["Unknown", "Bronze", "Silver", "Gold"], 
        index=0,
        key="plan_tier"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🆕 New Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.session_state.total_queries = 0
            st.rerun()
    with col_b:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown('<div class="sidebar-section">Quick Ask</div>', unsafe_allow_html=True)
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"eq_{q}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

    st.markdown('<div class="sidebar-section">Diagnostics</div>', unsafe_allow_html=True)
    if st.button("🔍 Fetch Last Trace", use_container_width=True):
        with st.spinner("Loading trace…"):
            async def _get_trace():
                async with httpx.AsyncClient() as c:
                    try:
                        r = await c.get(f"{API_BASE_URL}/session/{st.session_state.session_id}/trace")
                        return r.json()
                    except: return None
            td = asyncio.run(_get_trace())
            if td and "detail" not in td:
                st.json(td)
            else:
                st.info("Ask a question first.")

    # Stats
    st.markdown('<div class="sidebar-section">Stats</div>', unsafe_allow_html=True)
    st.metric("Queries this session", st.session_state.total_queries)

    st.markdown("""
    <div style='padding-top:24px; font-size:0.68rem; color:#334155; text-align:center;'>
        v1.0.0 · LangGraph · GPT-4o · LangSmith
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 6. MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════

# Hero header — shown only when no messages
if not st.session_state.messages:
    st.markdown(f"""
    <div class="main-hero">
        <span class="hero-icon">🛡️</span>
        <div class="hero-title">{APP_TITLE}</div>
        <div class="hero-sub">Ask about coverage, providers, pharmacy benefits, and plan comparisons.</div>
    </div>
    """, unsafe_allow_html=True)

    # Inline quick-start cards below hero
    st.markdown("<div style='margin-bottom:10px;font-size:0.8rem;color:#475569;'>Try asking:</div>",
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUERIES[:4]):
        with cols[i % 2]:
            if st.button(q, key=f"hero_{q}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()

else:
    # Compact status strip when conversation is active
    st.markdown(f"""
    <div class="status-strip">
        <div class="status-dot"></div>
        Backend connected &nbsp;·&nbsp; {st.session_state.total_queries} message{"s" if st.session_state.total_queries != 1 else ""} this session
    </div>
    """, unsafe_allow_html=True)

# ── Render Chat History ────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="bubble">{msg["content"]}</div>
            <div class="avatar">👤</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        intent   = msg.get("intent", "POLICY_QUESTION")
        meta     = INTENT_METADATA.get(intent, INTENT_METADATA["POLICY_QUESTION"])
        ts       = msg.get("timestamp", "")[:16].replace("T", " ") if msg.get("timestamp") else ""
        run_id   = msg.get("run_id")
        memories = msg.get("memories_used", [])
        n_mems   = len(memories) if memories else 0

        # Build the badge row
        langsmith_html = ""
        if run_id:
            langsmith_html = f'<a class="langsmith-link" href="https://smith.langchain.com" target="_blank">📊 LangSmith Trace</a>'

        mem_html = f'<span class="memory-chip">🧠 {n_mems} memory fact{"s" if n_mems!=1 else ""}</span>' if n_mems else ""

        # Escape HTML in the answer for safe rendering inside the bubble
        answer_safe = msg["content"].replace("<", "&lt;").replace(">", "&gt;")

        # Build a single flat HTML string with no newlines or indentations to prevent markdown parsing issues
        badge_row = f'<div class="intent-row">'
        badge_row += f'<span class="intent-badge" style="background:{meta["color"]};color:white;">{meta["icon"]} {intent.replace("_"," ")}</span>'
        if ts:
            badge_row += f'<span class="ts-badge">{ts}</span>'
        if langsmith_html:
            badge_row += langsmith_html
        if mem_html:
            badge_row += mem_html
        badge_row += '</div>'

        html_content = (
            f'<div class="msg-ai">'
            f'<div class="avatar">🛡️</div>'
            f'<div class="bubble-wrapper">'
            f'{badge_row}'
            f'<div class="bubble">{answer_safe}</div>'
            f'</div>'
            f'</div>'
        )

        st.markdown(html_content, unsafe_allow_html=True)

        # Pipeline trace — collapsible
        steps = msg.get("steps_log", [])
        if steps:
            with st.expander(f"⛓️ Pipeline trace · {len(steps)} steps"):
                st.markdown("""
                <div class="pipeline-steps">
                """, unsafe_allow_html=True)
                for step in steps:
                    st.markdown(f"<div class='pipeline-step'>{step}</div>",
                                unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='chat-divider'>", unsafe_allow_html=True)

# ── Chat Input ──────────────────────────────────────────────

query = st.chat_input("Ask about your health insurance coverage…")

if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query

if query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.status("🧠 Processing your query…", expanded=True) as status:
        status.write("📡 Sending to orchestrator…")
        result = asyncio.run(call_chat_api(query))

        if result:
            status.write(f"✅ Intent → **{result.get('intent','—')}**")
            steps = result.get("steps_log", [])
            if steps:
                status.write(f"⛓️ {len(steps)} pipeline steps completed")
            status.update(label="✅ Answer ready!", state="complete", expanded=False)

            st.session_state.messages.append({
                "role":         "ai",
                "content":      result["answer"],
                "intent":       result.get("intent", "POLICY_QUESTION"),
                "steps_log":    result.get("steps_log", []),
                "memories_used": result.get("memories_used", []),
                "run_id":       result.get("run_id"),
                "timestamp":    result.get("timestamp", datetime.now().isoformat()),
            })
            st.session_state.total_queries += 1
            st.rerun()
        else:
            status.update(label="❌ Backend error", state="error")
