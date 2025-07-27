#!/usr/bin/env python3
"""
FastAPI server for Dark.RL.
Provides an OpenAI-compliant chat completions endpoint and a custom learn endpoint.
"""
import asyncio
import json
import time
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

from rich.console import Console
from rich.panel import Panel

from dark.online_llm import AsyncOnlineLLM
from dark.sampling_params import SamplingParams

# Import TaskManager from root directory
import sys
sys.path.append('.')
from task_manager import TaskManager


# --- Globals ---
console = Console()
llm: Optional[AsyncOnlineLLM] = None
app = FastAPI(
    title="Dark.RL API",
    description="OpenAI-compliant API for Dark.RL models with online learning capabilities."
)

# --- Pydantic Models for API Validation ---

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    adapter: Optional[str] = None  # Custom parameter for LoRA adapter

class LearnRequest(BaseModel):
    messages: List[Dict[str, Any]]
    adapter: str = "default"
    steps: Optional[int] = 5
    lr: Optional[float] = 1e-4

class CreateTaskRequest(BaseModel):
    initial_prompt: str
    model: Optional[str] = "qwen2.5-vl"

class CreateTaskResponse(BaseModel):
    task_id: str

class MCPServer(BaseModel):
    id: str
    name: str
    description: str
    api_key_available: bool
    required_env: List[str]
    optional_env: List[str]

class MCPAction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ExecuteMCPActionRequest(BaseModel):
    action_name: str
    parameters: Dict[str, Any]

class ExecuteMCPActionResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    error: Optional[str] = None

# --- FastAPI Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize the AsyncOnlineLLM instance on server startup."""
    global llm
    console.print(Panel(
        "🧠 Initializing AI model...",
        title="🔄 MODEL LOADING",
        style="bold yellow"
    ))
    try:
        # We can specify model, lora_rank, etc. here
        llm = AsyncOnlineLLM(
            model="Qwen/Qwen3-8B",
            thinking_mode=True,
        )
        console.print(Panel(
            "✅ AI model loaded successfully!",
            title="🧠 MODEL READY",
            style="bold green"
        ))
    except Exception as e:
        console.print(Panel(
            f"❌ Failed to load AI model: {e}",
            title="🚨 MODEL LOADING ERROR",
            style="bold red"
        ))
        llm = None

# --- API Endpoints ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compliant chat completions endpoint."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM is not initialized")

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    # Convert Pydantic models to dictionaries for the LLM
    messages_for_llm = [msg.dict() for msg in request.messages]

    if request.stream:
        # Streaming response
        async def stream_generator():
            stream_id = f"chatcmpl-stream-{int(time.time())}"
            chunk_index = 0
            
            async for chunk in llm.chat_stream(
                msgs=messages_for_llm,
                adapter=request.adapter,
                sampling_params=sampling_params
            ):
                response_chunk = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(response_chunk)}\n\n"
                chunk_index += 1
            
            # Send final chunk with finish_reason
            final_chunk = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # Non-streaming response
        start_time = time.time()
        full_response = await llm.chat(
            msgs=messages_for_llm,
            adapter=request.adapter,
            sampling_params=sampling_params
        )
        end_time = time.time()
        
        response_id = f"chatcmpl-{int(start_time)}"
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(start_time),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                # Placeholder values for usage
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }

@app.post("/v1/learn")
async def learn_from_chat(request: LearnRequest):
    """Endpoint to trigger online learning for the model."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM is not initialized")

    try:
        await llm.learn(
            msgs=request.messages,
            adapter=request.adapter,
            steps=request.steps,
            lr=request.lr
        )
        return {"status": "success", "detail": f"Learning task for adapter '{request.adapter}' completed."}
    except Exception as e:
        console.print_exception()
        raise HTTPException(status_code=500, detail=f"Failed to learn: {str(e)}")

@app.get("/v1/mcp/servers", response_model=List[MCPServer])
async def list_mcp_servers(request: Request, query: Optional[str] = None):
    """Endpoint to list available MCP servers, with optional query."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM is not initialized")
    
    try:
        # The get_mcp_servers logic is now on the llm object.
        servers = await llm.get_mcp_servers(query if query else "")
        return servers
    except Exception as e:
        console.print_exception()
        raise HTTPException(status_code=500, detail=f"Failed to list MCP servers: {str(e)}")

@app.get("/v1/mcp/servers/{server_id}/actions", response_model=List[MCPAction])
async def get_mcp_server_actions(server_id: str, request: Request):
    """Endpoint to get available actions for a specific MCP server."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM is not initialized")
    
    try:
        actions = await llm.get_mcp_server_actions([server_id])
        if server_id not in actions:
            raise HTTPException(status_code=404, detail=f"Server {server_id} not found or has no actions")
        return actions[server_id]
    except Exception as e:
        console.print_exception()
        raise HTTPException(status_code=500, detail=f"Failed to get actions for server {server_id}: {str(e)}")

@app.post("/v1/mcp/servers/{server_id}/actions/execute", response_model=ExecuteMCPActionResponse)
async def execute_mcp_action(server_id: str, request: ExecuteMCPActionRequest, req: Request):
    """Endpoint to execute an action on a specific MCP server."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM is not initialized")
    
    try:
        result = await llm.execute_mcp_action(
            server_id,
            request.action_name,
            request.parameters
        )
        return result
    except Exception as e:
        console.print_exception()
        raise HTTPException(status_code=500, detail=f"Failed to execute action {request.action_name} on server {server_id}: {str(e)}")

@app.post("/v1/tasks", response_model=CreateTaskResponse)
async def create_task(request: CreateTaskRequest):
    """Endpoint to create a new task."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM is not initialized")
    
    try:
        # Assuming TaskManager is part of the llm instance
        task_manager = llm.task_manager 
        task_id = task_manager.create_task(request.initial_prompt, request.model)
        return CreateTaskResponse(task_id=task_id)
    except Exception as e:
        console.print_exception()
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

@app.get("/v1/task/{task_id}")
async def get_task(task_id: str):
    """Endpoint to get task data by ID."""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM is not initialized")
    
    task_manager = llm.task_manager
    task_data = task_manager.get_task(task_id)

    if not task_data or task_data.get('error'):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return task_data


async def main():
    """Main function to start the API server."""
    config = uvicorn.Config(app, host="localhost", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print(Panel("🛑 Server stopped by user.", title="👋 SHUTDOWN", style="bold yellow"))
    except Exception as e:
        console.print(Panel(f"❌ Server error: {e}", title="🚨 FATAL ERROR", style="bold red"))
        import traceback
        traceback.print_exc()
