import streamlit as st
import streamlit.components.v1 as components
import httpx
import uuid
import asyncio
from datetime import datetime
import pandas as pd
from config import API_BASE_URL, APP_TITLE, APP_SUBTITLE, THEME, EXAMPLE_QUERIES, INTENT_METADATA

# ══════════════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION & STYLING
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism UI
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    :root {{
        --primary: {THEME["primary"]};
        --secondary: {THEME["secondary"]};
        --bg: {THEME["background"]};
        --card-bg: {THEME["card_bg"]};
        --text: {THEME["text"]};
    }}

    .stApp {{
        background-color: var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }}

    /* Glassmorphism containers */
    .glass-card {{
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }}

    /* Chat Bubbles */
    .chat-bubble {{
        padding: 12px 18px;
        border-radius: 18px;
        margin-bottom: 10px;
        max-width: 80%;
        line-height: 1.5;
        position: relative;
    }}

    .user-bubble {{
        background: linear-gradient(135deg, var(--primary), #6366F1);
        color: white;
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }}

    .ai-bubble {{
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: var(--text);
        align-self: flex-start;
        border-bottom-left-radius: 4px;
    }}

    /* Intent Badges */
    .intent-badge {{
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        margin-bottom: 8px;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }}

    /* Hide default Streamlit elements for cleaner look */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .animate-fade-in {{
        animation: fadeIn 0.5s ease-out forwards;
    }}

    /* Source Citation Cards */
    .source-card {{
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 10px;
        margin-top: 8px;
        font-size: 0.85rem;
        border-left: 3px solid var(--secondary);
    }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 2. SESSION STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

# ══════════════════════════════════════════════════════════════
# 3. API WRAPPERS
# ══════════════════════════════════════════════════════════════

async def call_chat_api(query: str):
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "session_id": st.session_state.session_id,
            "query": query
        }
        try:
            response = await client.post(f"{API_BASE_URL}/chat", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error connecting to backend: {str(e)}")
            return None

async def get_health():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            return response.json()
        except:
            return None

async def get_sessions():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/sessions")
            return response.json()
        except:
            return []

async def get_chat_history(session_id: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/chat/history", params={"session_id": session_id})
            return response.json()
        except:
            return []

# ══════════════════════════════════════════════════════════════
# 4. SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"<h1 style='color: white; margin-bottom: 0;'>🛡️ {APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #94A3B8; font-size: 0.9rem;'>{APP_SUBTITLE}</p>", unsafe_allow_html=True)
    
    # Session Info
    st.markdown("### 🆔 Session")
    st.code(st.session_state.session_id[:13] + "...", language=None)
    if st.button("🆕 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.link_button("🚀 Open Developer Console", "http://localhost:8000/dev-console", use_container_width=True, help="Requires password '2002'")
    st.markdown("---")
    
    # Chat History
    st.markdown("### 💬 Chat History")
    sessions = asyncio.run(get_sessions())
    if sessions:
        for s in sessions[:5]: # Show last 5
            try:
                time_str = datetime.fromisoformat(s['updated_at']).strftime("%H:%M")
            except:
                time_str = "??:??"
            if st.button(f"📅 {time_str} | {s['id'][:8]}...", key=f"hist_{s['id']}", use_container_width=True):
                history = asyncio.run(get_chat_history(s['id']))
                st.session_state.session_id = s['id']
                st.session_state.messages = history
                st.rerun()
    else:
        st.caption("No recent chats")

    st.markdown("---")
    
    # Example Queries
    st.markdown("### 💡 Example Queries")
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

    st.markdown("---")
    
    # System Status
    st.markdown("### ⚡ System Status")
    health = None # This would be fetched
    col1, col2 = st.columns(2)
    col1.metric("Status", "Online", delta_color="normal")
    col2.metric("Latency", "~1.2s")

    if st.button("🔍 View Detailed Workflow Trace", use_container_width=True):
        with st.status("Fetching trace...", expanded=True) as status:
            async def get_trace():
                async with httpx.AsyncClient() as client:
                    try:
                        resp = await client.get(f"{API_BASE_URL}/session/{st.session_state.session_id}/trace")
                        return resp.json()
                    except:
                        return None
            
            trace_data = asyncio.run(get_trace())
            if trace_data and "detail" not in trace_data:
                status.update(label="✅ Trace retrieved", state="complete")
                st.json(trace_data)
            else:
                status.update(label="❌ No trace found", state="error")
                st.info("Ask a question first to generate a workflow trace.")
    
    if st.button("🖼️ View Workflow Graph", use_container_width=True):
        with st.status("Fetching graph...", expanded=True) as gstatus:
            async def get_graph():
                async with httpx.AsyncClient() as client:
                    try:
                        resp = await client.get(f"{API_BASE_URL}/session/{st.session_state.session_id}/graph")
                        return resp.content
                    except:
                        return None
            
            img_bytes = asyncio.run(get_graph())
            if img_bytes:
                gstatus.update(label="✅ Graph retrieved", state="complete")
                st.image(img_bytes, caption="Workflow Graph", use_column_width=True)
            else:
                gstatus.update(label="❌ No graph found", state="error")
                st.info("Ask a question first to generate a graph.")
    
    st.markdown("<div style='padding-top: 20px; font-size: 0.7rem; color: #64748b;'>v1.0.0 | Powered by LangGraph & GPT-4o</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 5. MAIN CONTENT AREA
# ══════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["💬 Assistant", "🛠️ Developer Console"])

# ── TAB 1: ASSISTANT ──────────────────────────────────────────
with tab1:
    # Header
    st.markdown(f"""
    <div class='glass-card animate-fade-in'>
        <h2 style='margin-top: 0; color: white;'>Welcome to {APP_TITLE}</h2>
        <p style='color: #cbd5e1;'>Ask me anything about your health insurance coverage, providers, or pharmacy benefits.</p>
    </div>
    """, unsafe_allow_html=True)

    # Display Chat History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble user-bubble animate-fade-in'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            intent = msg.get("intent", "POLICY_QUESTION")
            meta = INTENT_METADATA.get(intent, INTENT_METADATA["POLICY_QUESTION"])
            st.markdown(f"""
            <div class='chat-bubble ai-bubble animate-fade-in'>
                <div class='intent-badge' style='background: {meta["color"]}; color: white;'>
                    {meta["icon"]} {intent.replace("_", " ")}
                </div>
                <div style='margin-bottom: 10px;'>{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            if msg.get("steps_log"):
                with st.expander("🔍 Retrieval Steps"):
                    for step in msg["steps_log"]: st.write(f"  {step}")

    # Input Area
    query = st.chat_input("How can I help you today?")
    if "pending_query" in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query

    if query:
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.status("🧠 Thinking...", expanded=True) as status:
            # Call API
            result = asyncio.run(call_chat_api(query))
            
            if result:
                status.update(label="✅ Answer found!", state="complete", expanded=False)
                # Add AI message to session state
                ai_message = {
                    "role": "ai",
                    "content": result["answer"],
                    "intent": result["intent"],
                    "steps_log": result["steps_log"]
                }
                st.session_state.messages.append(ai_message)
                st.rerun()
            else:
                status.update(label="❌ Error generating answer", state="error")

# ── TAB 2: DEVELOPER CONSOLE ──────────────────────────────────
with tab2:
    st.markdown("### 🛠️ Live Agent Debugger")
    st.info("Enter a query below to see the internal agentic flow in real-time.")
    
    dev_query = st.text_input("Debug Query:", placeholder="e.g. Compare Bronze and Gold deductibles")
    
    if st.button("🚀 Run Live Trace", type="primary"):
        if dev_query:
            # Container for the live diagram
            diagram_container = st.empty()
            log_container = st.container()
            
            with st.status("📡 Initializing stream...", expanded=True) as status:
                try:
                    import json
                    
                    async def run_dev_stream():
                        async with httpx.AsyncClient(timeout=None) as client:
                            async with client.stream("GET", f"{API_BASE_URL}/chat/stream", params={"session_id": st.session_state.session_id, "query": dev_query}) as response:
                                async for line in response.aiter_lines():
                                    if line.startswith("data: "):
                                        data_str = line[6:]
                                        if data_str == "[DONE]":
                                            break
                                        
                                        data = json.loads(data_str)
                                        node = data["node"]
                                        steps = data["steps_log"]
                                        
                                        status.update(label=f"⛓️ Node Finished: {node}", state="running")
                                        
                                        # Update diagram in real-time
                                        # We'll fetch the diagram from the backend for the current state
                                        # Or better, just show the steps for now to ensure speed
                                        with log_container:
                                            st.write(f"**[{node}]** {steps[-1] if steps else ''}")
                                        
                                        # After synthesis, we can show the final diagram
                                        if node == "synthesize":
                                            status.update(label="✅ Workflow Complete", state="complete")
                                            # Fetch final diagram
                                            resp = await client.get(f"{API_BASE_URL}/session/{st.session_state.session_id}/graph")
                                            if resp.status_code == 200:
                                                diagram_container.image(resp.content, caption="Final Workflow Graph")

                    asyncio.run(run_dev_stream())
                except Exception as e:
                    st.error(f"Stream error: {str(e)}")
        else:
            st.warning("Please enter a query first.")

# Footer spacer
st.markdown("<br><br>", unsafe_allow_html=True)
