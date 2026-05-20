import os
import sys
import time
import json
import asyncio
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
import subprocess
import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse as StarletteStreamingResponse

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import (
    ChatRequest, ChatResponse, HealthResponse,
    SessionHistoryResponse, ChatMessage, ErrorResponse,
    WorkflowTraceResponse, WorkflowStep,
    WorkflowDiagramResponse,
    MemoryFact, MemoryResponse,
)
from orchestration.tools import preload_retrievers
from orchestration.memory import get_all_memories


from backend.session_manager import manager
from fastapi.responses import StreamingResponse
import io
from graphviz import Digraph

# Setup logging
logger.add("logs/backend.log", rotation="500 MB", level="INFO")

# Detect if running in monolith mode (e.g. on Hugging Face Spaces)
IS_MONOLITH = os.getenv("RUN_MONOLITH", "false").lower() == "true" or os.getenv("PORT") == "7860"

app = FastAPI(
    title="Health Insurance AI Copilot API",
    description="Backend API for the RAG-based Health Insurance Assistant",
    version="1.0.0"
)

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Proxy Middleware for Single-Port Deployment (Hugging Face) ──
if IS_MONOLITH and os.getenv("USE_NGINX", "false").lower() != "true":
    class StreamlitProxyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # Let FastAPI handle these routes directly
            if (request.url.path.startswith("/chat") or 
                request.url.path.startswith("/session") or 
                request.url.path.startswith("/health") or 
                request.url.path.startswith("/dev-console") or
                request.url.path.startswith("/static")):
                return await call_next(request)
            
            # Everything else goes to Streamlit (running on port 8501 internally)
            target_url = f"http://localhost:8501{request.url.path}"
            if request.url.query:
                target_url += f"?{request.url.query}"
                
            async with httpx.AsyncClient() as client:
                try:
                    proxy_req = client.build_request(
                        request.method,
                        target_url,
                        headers=request.headers.raw,
                        content=await request.body()
                    )
                    proxy_res = await client.send(proxy_req, stream=True)
                    return StarletteStreamingResponse(
                        proxy_res.aiter_raw(),
                        status_code=proxy_res.status_code,
                        headers=proxy_res.headers
                    )
                except Exception as e:
                    logger.error(f"Proxy error: {str(e)}")
                    return await call_next(request)

    app.add_middleware(StreamlitProxyMiddleware)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing retrieval pipeline preload synchronously...")
    preload_retrievers()
    
    if IS_MONOLITH:
        logger.info("Starting Streamlit frontend sidecar on port 8501...")
        subprocess.Popen([
            "streamlit", "run", "frontend/app.py", 
            "--server.port", "8501", 
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ])
        logger.info("Backend and Frontend sidecar are fully ready.")
    else:
        logger.info("Backend is fully ready.")

# Mount static frontend files
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

@app.get("/dev-console")
async def dev_console():
    """Serve the beautiful live Developer Console."""
    html_path = os.path.join(_frontend_dir, "dev_console.html")
    return FileResponse(html_path, media_type="text/html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the backend and its components."""
    # Simplified health check
    components = {
        "openai_api": "ok" if os.getenv("OPENAI_API_KEY") else "error",
        "vector_store": "ok", # In a real app, check Chroma connectivity
        "knowledge_graph": "ok"
    }
    
    status = "healthy"
    if any(v == "error" for v in components.values()):
        status = "degraded"
        
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        components=components,
        active_sessions=manager.get_active_session_count()
    )

@app.post("/chat", response_model=ChatResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def chat(request: ChatRequest):
    """Process a chat query for a specific session."""
    try:
        logger.info(f"Received query for session {request.session_id}: {request.query[:50]}...")
        
        # Get or create orchestrator for this session
        orch = manager.get_orchestrator(request.session_id)
        
        # Execute query using the detailed method we added
        start_time = time.time()
        
        final_query = request.query
        if request.plan_tier and request.plan_tier.lower() != "unknown":
            final_query = f"I am on the {request.plan_tier} plan. {request.query}"
            
        result = orch.ask_detailed(final_query)
        duration = time.time() - start_time
        
        logger.info(f"Query processed in {duration:.2f}s for session {request.session_id}")
        
        return ChatResponse(
            session_id=request.session_id,
            query=request.query,
            answer=result["answer"],
            intent=result["intent"],
            steps_log=result["steps_log"],
            memories_used=result.get("memories_used", []),
            timestamp=datetime.now().isoformat(),
            run_id=result.get("run_id"),   # LangSmith trace UUID (None if tracing off)
        )
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=ErrorResponse(error="Internal Server Error", detail=str(e), session_id=request.session_id).dict()
        )

@app.get("/chat/stream")
async def chat_stream(session_id: str, query: str, plan_tier: str = "Unknown"):
    """
    Streamed version of the chat process for the Developer Console.
    Emits granular JSON chunks: startup nodes, then per-step events within each LangGraph node.
    """
    import threading
    orch = manager.get_orchestrator(session_id)
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    final_query = query
    if plan_tier and plan_tier.lower() != "unknown":
        final_query = f"I am on the {plan_tier} plan. {query}"

    def run_stream():
        try:
            for event in orch.stream_detailed(final_query):
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "event", "data": event})
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "error": e})

    # Start the LangGraph execution in a background thread so it doesn't block the main event loop
    thread = threading.Thread(target=run_stream, daemon=True)
    thread.start()

    async def event_generator():
        def emit(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        # ── Startup handshake events ──────────────────────────────
        yield emit({"type": "node_start", "node": "user",         "msg": query})
        await asyncio.sleep(0.25)
        yield emit({"type": "node_start", "node": "fastapi",      "msg": "POST /chat/stream received"})
        await asyncio.sleep(0.25)
        yield emit({"type": "node_start", "node": "orchestrator",  "msg": "Building LangGraph state…"})
        await asyncio.sleep(0.35)

        prev_steps: list[str] = []
        final_answer = ""
        last_intent  = ""

        # ── Stream LangGraph events ───────────────────────────────
        try:
            while True:
                try:
                    # Wait for 2.0s for the next event, if none, send keepalive
                    item = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    # SSE keepalive comment to keep the connection alive through reverse proxies
                    yield ": keepalive\n\n"
                    continue

                if item["type"] == "done":
                    break
                elif item["type"] == "error":
                    raise item["error"]

                event = item["data"]
                node  = event["node"]
                state = event["state"]
                current_steps = state.get("steps_log", [])
                new_steps     = current_steps[len(prev_steps):]   # only new lines
                intent        = state.get("intent", last_intent)
                last_intent   = intent

                # Tell the frontend the LangGraph node just started
                yield emit({"type": "node_start", "node": node, "intent": intent, "msg": f"Node '{node}' executing…"})
                await asyncio.sleep(0.15)

                # Emit individual sub-steps (each line from steps_log) with a small delay
                for step in new_steps:
                    yield emit({"type": "substep", "node": node, "intent": intent, "step": step, "all_steps": current_steps})
                    await asyncio.sleep(0.05)   # minimal pause for smooth UI without blocking stream

                # Signal node completion
                final_answer = state.get("answer", "")
                yield emit({
                    "type":   "node_done",
                    "node":   node,
                    "intent": intent,
                    "steps":  current_steps,
                    "answer": final_answer,
                })

                # Persist completed state for trace/graph endpoints
                if node == "synthesize":
                    if len(orch.chat_history) == 0 or orch.chat_history[-1] != ("ai", final_answer):
                        orch.chat_history.append(("human", query))
                        orch.chat_history.append(("ai",    final_answer))
                    orch.last_detailed_result = {
                        "query":             query,
                        "answer":            final_answer,
                        "intent":            intent,
                        "steps_log":         current_steps,
                        "retrieved_context": state.get("retrieved_context", ""),
                    }

                prev_steps = current_steps
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in chat_stream: {str(e)}")
            yield emit({"type": "error", "msg": f"Backend Error: {str(e)}"})

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_history(session_id: str):
    """Retrieve the chat history for a specific session."""
    try:
        orch = manager.get_orchestrator(session_id)
        history = []
        for role, content in orch.chat_history:
            history.append(ChatMessage(role=role, content=content))
            
        return SessionHistoryResponse(
            session_id=session_id,
            history=history,
            message_count=len(history)
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or error retrieving history")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session's history, state, and in-memory Mem0 facts."""
    manager.clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id} cleared (including memory)"}


@app.get("/memory/{session_id}", response_model=MemoryResponse)
async def get_memory(session_id: str):
    """
    Inspect all facts Mem0 has stored for this session.
    Useful for debugging — shows exactly what the agent remembers.
    Note: ephemeral — resets when the server restarts.
    """
    try:
        mem = manager.get_memory(session_id)
        if mem is None:
            return MemoryResponse(
                session_id=session_id,
                memory_count=0,
                facts=[],
                note="Session not found or memory disabled"
            )
        raw_facts = get_all_memories(mem)
        facts = [
            MemoryFact(
                id=str(f.get("id", "")),
                memory=f.get("memory", ""),
                created_at=str(f.get("created_at", ""))
            )
            for f in raw_facts
        ]
        return MemoryResponse(
            session_id=session_id,
            memory_count=len(facts),
            facts=facts
        )
    except Exception as e:
        logger.error(f"Error fetching memory for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """
    Wipe all Mem0 facts for this session without clearing the full session.
    The session and chat history remain intact.
    """
    try:
        mem = manager.get_memory(session_id)
        if mem is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        # Re-create a fresh memory instance for the session
        from orchestration.memory import create_session_memory
        orch = manager.get_orchestrator(session_id)
        new_mem = create_session_memory()
        orch._mem = new_mem
        orch._search_memories = lambda query: __import__('orchestration.memory', fromlist=['search_memories']).search_memories(new_mem, query)
        orch._add_memory       = lambda q, a:  __import__('orchestration.memory', fromlist=['add_memory']).add_memory(new_mem, q, a)
        return {"status": "success", "message": f"Memory cleared for session {session_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing memory for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper to turn steps into a Graphviz diagram (PNG bytes)
def _steps_to_graphviz(query: str, intent: str, steps: list[str]) -> io.BytesIO:
    """Generate a PNG image of the workflow using Graphviz.
    Nodes: FastAPI, Orchestrator, Intent, Retrieval, Synthesis, Answer.
    Detailed steps are added as chained nodes attached to Retrieval.
    """
    dot = Digraph(comment='Workflow')
    dot.attr(rankdir='LR')
    # Core nodes
    dot.node('A', 'FastAPI Endpoint')
    dot.node('B', 'Orchestrator')
    dot.node('C', 'Intent Classification')
    dot.node('D', 'Retrieval Pipeline')
    dot.node('E', 'Synthesis Agent')
    dot.node('F', 'Answer')
    # Edges
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    # Detailed steps subgraph
    if steps:
        with dot.subgraph(name='cluster_details') as c:
            c.attr(label='Retrieval Details')
            prev = None
            for i, s in enumerate(steps, start=1):
                node_id = f'S{i}'
                safe = s.replace('"', '\\"')
                c.node(node_id, safe, shape='box')
                if i == 1:
                    dot.edge('D', node_id)
                else:
                    c.edge(prev, node_id)
                prev = node_id
    # Render to PNG in memory
    png_bytes = dot.pipe(format='png')
    return io.BytesIO(png_bytes)

def _steps_to_mermaid(query: str, intent: str, steps: list[str]) -> str:
    """Generate a simple mermaid diagram describing the workflow.
    The diagram will have the form:
        FastAPI --> Orchestrator --> Intent --> Retrieval --> Synthesis
    and then each step from steps will be added as a sub‑node.
    """
    lines = ["graph LR"]
    lines.append('    A[FastAPI Endpoint] --> B[Orchestrator]')
    lines.append('    B --> C[Intent Classification]')
    lines.append('    C --> D[Retrieval Pipeline]')
    lines.append('    D --> E[Synthesis Agent]')
    lines.append('    E --> F[Answer]')
    # Add detailed steps as a vertical sub‑graph attached to D
    if steps:
        lines.append('    subgraph Details [Retrieval Details]')
        for i, s in enumerate(steps, start=1):
            # sanitize characters that break mermaid syntax
            safe = s.replace('"', '\\"')
            node_id = f"S{i}"
            lines.append(f'        {node_id}["{safe}"]')
            if i == 1:
                lines.append(f'        D --> {node_id}')
            else:
                lines.append(f'        S{i-1} --> {node_id}')
        lines.append('    end')
    return "\n".join(lines)

@app.get("/session/{session_id}/diagram", response_model=WorkflowDiagramResponse)
async def get_diagram(session_id: str):
    """Return a Mermaid diagram visualising the most recent query workflow for a session."""
    try:
        orch = manager.get_orchestrator(session_id)
        result = orch.last_detailed_result
        if not result:
            raise HTTPException(status_code=404, detail="No trace found for this session. Ask a question first.")
        diagram = _steps_to_mermaid(
            query=result.get("query", ""),
            intent=result.get("intent", ""),
            steps=result.get("steps_log", []),
        )
        return WorkflowDiagramResponse(
            session_id=session_id,
            query=result.get("query", ""),
            intent=result.get("intent", ""),
            diagram=diagram,
            steps_log=result.get("steps_log", []),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating diagram: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/session/{session_id}/graph", response_model=None)
async def get_graph(session_id: str):
    """Return a PNG image visualising the latest query workflow for a session."""
    try:
        orch = manager.get_orchestrator(session_id)
        result = orch.last_detailed_result
        if not result:
            raise HTTPException(status_code=404, detail="No trace found for this session. Ask a question first.")
        img_io = _steps_to_graphviz(
            query=result.get("query", ""),
            intent=result.get("intent", ""),
            steps=result.get("steps_log", []),
        )
        return StreamingResponse(img_io, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/session/{session_id}/trace", response_model=WorkflowTraceResponse)
async def get_trace(session_id: str):
    """Retrieve a detailed workflow trace for the last query in a session."""
    try:
        orch = manager.get_orchestrator(session_id)
        if not orch.last_detailed_result:
            raise HTTPException(status_code=404, detail="No trace found for this session. Ask a question first.")
            
        res = orch.last_detailed_result
        
        # Map the flat steps_log to a more structured workflow if possible
        # This is a bit heuristic but provides the requested "agent1 -> agent2" feel
        steps = []
        for i, s in enumerate(res["steps_log"]):
            name = "Process Step"
            if "Intent" in s: name = "Intent Classification"
            elif "Graph" in s: name = "Knowledge Graph Retrieval"
            elif "Hybrid" in s or "document" in s: name = "Policy Retrieval"
            elif "synthesized" in s: name = "LLM Synthesis"
            
            steps.append(WorkflowStep(
                step_id=f"step_{i+1}",
                name=name,
                description=s
            ))
            
        return WorkflowTraceResponse(
            session_id=session_id,
            query=res["query"],
            intent=res["intent"],
            steps=steps,
            full_trace=res["steps_log"],
            retrieved_context_preview=res["retrieved_context"][:500] + "..." if len(res["retrieved_context"]) > 500 else res["retrieved_context"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving trace: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
