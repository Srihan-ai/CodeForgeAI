"""
CodeForge AI - Intelligent Code Generation Platform
====================================================
A streamlined multi-agent code generation system powered by LangGraph.

Features:
- Multi-language support (Python, Java, C++)
- Self-correcting code generation
- Real-time streaming
- Built-in guardrails
"""

import os
import json
import time
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import agent, CrewState, extract_task_intent, generate_artifact_filename
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 CodeForge AI starting...")
    yield
    logger.info("👋 CodeForge AI shutting down...")

app = FastAPI(
    title="CodeForge AI",
    description="Intelligent multi-agent code generation platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response Models
class CodeRequest(BaseModel):
    task: str = Field(..., description="Description of code to generate")
    language: str = Field(default="python", description="Target language")
    max_iterations: int = Field(default=3, ge=1, le=5)

class CodeResponse(BaseModel):
    success: bool
    code: Optional[str] = None
    filename: Optional[str] = None
    report: Optional[str] = None
    iterations: int = 0
    error: Optional[str] = None

# Routes
@app.get("/")
async def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"status": "ok", "service": "CodeForge AI", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "CodeForge AI",
        "version": "1.0.0"
    }

@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """Generate code using the multi-agent workflow"""
    try:
        logger.info(f"Generating code: {request.task[:50]}...")
        
        task_intent = extract_task_intent(request.task)
        filename = generate_artifact_filename(task_intent, request.language)
        
        initial_state: CrewState = {
            "messages": [HumanMessage(content=request.task)],
            "task_specification": request.task,
            "task_intent": task_intent,
            "language": request.language,
            "target_language": request.language,
            "code": None,
            "filename": filename,
            "report": None,
            "execution_success": False,
            "iterations": 0,
            "max_iterations": request.max_iterations,
            "hitl_enabled": False
        }
        
        # Add thread_id for checkpointing
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = agent.invoke(initial_state, config)
        
        return CodeResponse(
            success=result.get("execution_success", False),
            code=result.get("code"),
            filename=filename,
            report=result.get("report"),
            iterations=result.get("iterations", 0),
            error=None if result.get("execution_success") else "Tests failed"
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Generation failed", "message": str(e)}
        )

@app.get("/stream")
async def stream_generation(
    task: str = Query(...),
    language: str = Query("python"),
    max_iterations: int = Query(3)
):
    """Stream real-time generation events"""
    async def event_generator():
        try:
            task_intent = extract_task_intent(task)
            filename = generate_artifact_filename(task_intent, language)
            
            yield f"data: {json.dumps({'event': 'start', 'message': 'Starting generation...'})}\n\n"
            await asyncio.sleep(0.5)
            
            initial_state: CrewState = {
                "messages": [HumanMessage(content=task)],
                "task_specification": task,
                "task_intent": task_intent,
                "language": language,
                "target_language": language,
                "code": None,
                "filename": filename,
                "report": None,
                "execution_success": False,
                "iterations": 0,
                "max_iterations": max_iterations,
                "hitl_enabled": False
            }
            
            # Add thread_id for checkpointing
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            result = agent.invoke(initial_state, config)
            
            yield f"data: {json.dumps({'event': 'complete', 'code': result.get('code'), 'success': result.get('execution_success')})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
