#!/usr/bin/env python3
"""
WebSocket server for Dark.RL frontend using real AsyncOnlineLLM.
Accepts OpenAI-format requests and returns streaming responses from actual models.
Now includes task management with SQLite persistence and real MCP server integration.
"""

import asyncio
import json
import logging
import websockets
import os
import re
import time
from typing import Dict, Any, List
from pathlib import Path
import openai
import datetime

# Rich imports for beautiful logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import box

# Import Dark.RL components
from dark.online_llm import AsyncOnlineLLM, BatchConfig
from dark.sampling_params import SamplingParams
from task_manager import TaskManager
from mcp_use import MCPClient

# Configure logging with DEBUG level for detailed troubleshooting
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Rich console for beautiful output
console = Console()

# Global LLM server instance (loaded at startup)
global_llm_server = None

async def log_websocket_send(websocket, data_dict, prefix="SEND"):
    """Log and send WebSocket message"""
    json_str = json.dumps(data_dict)
    logger.debug(f"🔥 {prefix}: {json_str[:200]}...")
    await websocket.send(json_str)

def load_local_api_keys() -> Dict[str, str]:
    """Load API keys from local ~/.agentsea/api_keys.json file"""
    try:
        agentsea_dir = Path.home() / '.agentsea'
        api_keys_file = agentsea_dir / 'api_keys.json'
        
        if not api_keys_file.exists():
            return {}
        
        with open(api_keys_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading local API keys: {e}")
        return {}

def get_api_key(env_var: str) -> str:
    """Get API key from environment variables or local file"""
    # First check environment variables
    env_value = os.getenv(env_var, "")
    if env_value:
        return env_value
    
    # Then check local file
    local_keys = load_local_api_keys()
    return local_keys.get(env_var, "")

class DarkRLLLMServer:
    """Dark.RL LLM server that provides real AI responses with task management"""
    
    def __init__(self):
        self.llm = None
        self.models = {
            "qwen3": "Qwen/Qwen3-8B",
            "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct", 
            "custom": "Qwen/Qwen3-8B" 
        }
        self.current_model = None
        self.task_manager = TaskManager()
        
        # Initialize OpenAI client
        self.openai_client = None
        self._init_openai_client()
        
        # Real MCP server configurations
        self.mcp_servers = [
            {
                "id": "playwright",
                "name": "Playwright",
                "description": "Web automation and testing",
                "config": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"]
                },
                "required_env": [],
                "optional_env": []
            },
            {
                "id": "filesystem",
                "name": "File System",
                "description": "File and directory operations",
                "config": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/ubuntu/dark.rl"]
                },
                "required_env": [],
                "optional_env": []
            },
            {
                "id": "sequential-thinking",
                "name": "Sequential Thinking",
                "description": "Sequential reasoning and thought processes",
                "config": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
                },
                "required_env": [],
                "optional_env": []
            },
            {
                "id": "firecrawl",
                "name": "Firecrawl",
                "description": "Web scraping and content extraction",
                "config": {
                    "command": "npx",
                    "args": ["-y", "firecrawl-mcp"],
                    "env": {
                        "FIRECRAWL_API_KEY": ""
                    }
                },
                "required_env": ["FIRECRAWL_API_KEY"],
                "optional_env": []
            },
            {
                "id": "postgres",
                "name": "PostgreSQL",
                "description": "Database operations",
                "config": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-postgres"],
                    "env": {
                        "POSTGRES_CONNECTION_STRING": ""
                    }
                },
                "required_env": ["POSTGRES_CONNECTION_STRING"],
                "optional_env": []
            },
            {
                "id": "github",
                "name": "GitHub",
                "description": "Git repository management",
                "config": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {
                        "GITHUB_PERSONAL_ACCESS_TOKEN": ""
                    }
                },
                "required_env": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
                "optional_env": []
            },
            {
                "id": "slack",
                "name": "Slack",
                "description": "Team communication",
                "config": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "slack-mcp-server@latest",
                        "--transport",
                        "stdio"
                    ],
                    "env": {
                        "SLACK_MCP_XOXP_TOKEN": ""
                    }
                },
                "required_env": ["SLACK_MCP_XOXP_TOKEN"],
                "optional_env": []
            },
            {
                "id": "reddit",
                "name": "Reddit",
                "description": "Reddit operations",
                "config": {
                    "command": "uv",
                    "args": [
                        "run",
                        "python",
                        "-m",
                        "reddit_mcp.server"
                    ],
                    "env": {
                        "REDDIT_CLIENT_ID": "your_client_id",
                        "REDDIT_CLIENT_SECRET": "your_client_secret",
                        "REDDIT_USERNAME": "your_username",
                        "REDDIT_PASSWORD": "your_password"
                    }
                },
                "required_env": ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
                "optional_env": ["REDDIT_USERNAME", "REDDIT_PASSWORD"]
            }
        ]
        
        # MCP client connections cache
        self.mcp_clients = {}
        self.mcp_sessions = {}
        
        # MCP server actions cache
        self.mcp_actions_cache = {}
    
    def _init_openai_client(self):
        """Initialize OpenAI client with API key"""
        api_key = get_api_key("OPENAI_API_KEY")
        if api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=api_key)
            logger.info("✅ OpenAI client initialized")
        else:
            logger.warning("⚠️ OpenAI API key not found - dual model comparison disabled")
    
    async def stream_gpt_response(self, messages: List[dict], model: str = "gpt-4.1", temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """Stream response from GPT-4.1"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized - API key required")
        
        try:
            # Convert messages to OpenAI format
            openai_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role in ['system', 'user', 'assistant']:
                    openai_messages.append({
                        "role": role,
                        "content": content
                    })
            
            # Create streaming response
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            accumulated_response = ""
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    accumulated_response += chunk.choices[0].delta.content
            
            return accumulated_response
            
        except Exception as e:
            logger.error(f"Error streaming GPT response: {e}")
            return f"[GPT Error: {str(e)}]"
    
    async def stream_dual_response(self, websocket, messages: list, model: str = "qwen3", temperature: float = 0.7, max_tokens: int = 2048, task_id: str = None):
        """Stream responses from both local model and GPT-4.1 simultaneously with simplified state management"""
        
        console.print("🔥 BACKEND: Starting dual response")
        console.print(f"🔥   - model: {model}")
        console.print(f"🔥   - task_id: {task_id}")
        console.print(f"🔥   - messages count: {len(messages)}")
        console.print(f"🔥   - last message: {messages[-1]['content'][:100] if messages else 'None'}...")
        
        # Generate unique session ID for this dual response
        session_id = int(time.time() * 1000)  # millisecond timestamp as session ID
        console.print(f"🔥   - session_id: {session_id}")
        
        # Set task state to streaming_dual
        if task_id:
            self.task_manager.set_task_state(task_id, 'streaming_dual')
        
        # Initialize empty dual responses in database
        if task_id:
            self.task_manager.set_pending_dual_responses(
                task_id=task_id,
                local_response="",
                gpt_response="",
                local_model=model,
                gpt_model="gpt-4.1",
                session_id=session_id
            )
        
        # Send initial dual response header
        dual_start = {
            "type": "dual_response_start",
            "local_model": model,
            "gpt_model": "gpt-4.1", 
            "session_id": session_id,
            "task_id": task_id,
            "is_continuation": task_id is not None  # Flag continuation when we have a task_id
        }
        console.print(f"🔥 BACKEND: About to send dual_response_start: {json.dumps(dual_start)}")
        await websocket.send(json.dumps(dual_start))
        console.print("🔥 BACKEND: Successfully sent dual_response_start")
        
        # Add a small delay to ensure message is sent before chunks start
        await asyncio.sleep(0.01)
        
        # Create tasks for both models
        local_task = asyncio.create_task(self._stream_local_response(websocket, messages, model, temperature, max_tokens, task_id, session_id))
        gpt_task = asyncio.create_task(self._stream_gpt_response(websocket, messages, temperature, max_tokens, session_id, task_id))
        
        # Wait for both to complete
        local_response, gpt_response = await asyncio.gather(local_task, gpt_task, return_exceptions=True)
        
        # Update task state to awaiting_dual_selection
        if task_id:
            self.task_manager.set_task_state(task_id, 'awaiting_dual_selection')
        
        # Send completion marker
        dual_complete = {
            "type": "dual_response_complete",
            "local_finished": not isinstance(local_response, Exception),
            "gpt_finished": not isinstance(gpt_response, Exception),
            "session_id": session_id,
            "task_id": task_id
        }
        await websocket.send(json.dumps(dual_complete))
        console.print("🔥 BACKEND: Sent dual_response_complete")
        
        return local_response, gpt_response

    async def stream_dual_action_response(self, websocket, messages: list, mentioned_servers: List[str], model: str = "qwen3", temperature: float = 0.7, max_tokens: int = 2048, task_id: str = None):
        """Stream responses from both local model (with tools) and GPT-4.1 (with tools) simultaneously"""
        
        console.print("🔥 BACKEND: Starting dual action response")
        console.print(f"🔥   - model: {model}")
        console.print(f"🔥   - task_id: {task_id}")
        console.print(f"🔥   - mentioned_servers: {mentioned_servers}")
        console.print(f"🔥   - messages count: {len(messages)}")
        console.print(f"🔥   - last message: {messages[-1]['content'][:100] if messages else 'None'}...")
        
        # Generate unique session ID for this dual response
        session_id = int(time.time() * 1000)  # millisecond timestamp as session ID
        console.print(f"🔥   - session_id: {session_id}")
        
        # Send initial dual response header
        dual_start = {
            "type": "dual_response_start",
            "local_model": f"{model} (with tools)",
            "gpt_model": "gpt-4.1 (reasoning only)",
            "session_id": session_id,
            "task_id": task_id,
            "is_continuation": task_id is not None  # Flag continuation when we have a task_id
        }
        console.print(f"🔥 BACKEND: About to send dual_response_start: {json.dumps(dual_start)}")
        await websocket.send(json.dumps(dual_start))
        console.print("🔥 BACKEND: Successfully sent dual_response_start for action mode")
        
        # Add a small delay to ensure message is sent before chunks start
        await asyncio.sleep(0.01)
        
        # Build tools schema for mentioned servers - BOTH models get the same context
        tools = await self.build_tools_schema(mentioned_servers)
        
        if not tools:
            # No tools found, fall back to dual chat mode
            logger.warning(f"No tools found for servers: {mentioned_servers}")
            return await self.stream_dual_response(websocket, messages, model, temperature, max_tokens, task_id)
        
        # Build Qwen chat messages with tools - THIS IS THE GROUND TRUTH PROMPT
        qwen_chat_messages = self.build_qwen_chat_messages(messages, tools)
        
        # Extract the actual prompt that will be sent to the local model
        if hasattr(self.llm, '_messages_to_prompt'):
            await self.initialize_model(model)
            actual_prompt = self.llm._messages_to_prompt(qwen_chat_messages)
        else:
            # Fallback: manually construct a similar prompt
            actual_prompt = ""
            for msg in qwen_chat_messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    actual_prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == 'user':
                    actual_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == 'assistant':
                    actual_prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            actual_prompt += "<|im_start|>assistant\n"
        
        # Create tasks for both models
        console.print(f"🔥 BACKEND: Creating tasks for both models...")
        console.print(f"🔥   - About to create local_task")
        
        # Local model gets full tool context
        local_task = asyncio.create_task(self._stream_local_action_response(websocket, messages, mentioned_servers, model, temperature, max_tokens, task_id, session_id))
        console.print(f"🔥   - Local task created: {local_task}")
        console.print(f"🔥   - About to create gpt_task")
        
        # GPT-4.1 gets the EXACT SAME prompt as the local model for fair comparison
        gpt_task = asyncio.create_task(self._stream_gpt_with_same_prompt(websocket, actual_prompt, temperature, max_tokens, session_id))
        console.print(f"🔥   - GPT task created: {gpt_task}")
        console.print(f"🔥   - Both tasks created, waiting for completion...")
        console.print(f"🔥   - Local task done: {local_task.done()}")
        console.print(f"🔥   - GPT task done: {gpt_task.done()}")
        
        # Wait for both to complete with timeout
        try:
            local_response, gpt_response = await asyncio.wait_for(
                asyncio.gather(local_task, gpt_task, return_exceptions=True),
                timeout=120.0  # 2 minute timeout for both tasks
            )
            console.print(f"🔥   - Tasks completed successfully")
        except asyncio.TimeoutError:
            console.print(f"🔥   - Tasks timed out after 2 minutes")
            logger.error(f"Dual response tasks timeout for session {session_id}")
            
            # Cancel tasks if they're still running
            if not local_task.done():
                local_task.cancel()
            if not gpt_task.done():
                gpt_task.cancel()
            
            # Return timeout responses
            local_response = "[Local timeout: Task took too long to complete]"
            gpt_response = "[GPT timeout: Task took too long to complete]"
        console.print(f"🔥   - Tasks completed:")
        console.print(f"🔥     - local_response type: {type(local_response)}")
        console.print(f"🔥     - gpt_response type: {type(gpt_response)}")
        if isinstance(local_response, Exception):
            console.print(f"🔥     - local_response ERROR: {local_response}")
        if isinstance(gpt_response, Exception):
            console.print(f"🔥     - gpt_response ERROR: {gpt_response}")
        
        # Send completion marker
        dual_complete = {
            "type": "dual_response_complete",
            "local_finished": not isinstance(local_response, Exception),
            "gpt_finished": not isinstance(gpt_response, Exception),
            "session_id": session_id
        }
        await websocket.send(json.dumps(dual_complete))
        console.print("🔥 BACKEND: Sent dual_response_complete for action mode")
        
        # IMPORTANT: Don't auto-continue after dual response - wait for user to select which response they prefer
        # This prevents the infinite loop
        
        return local_response, gpt_response

    async def _stream_local_response(self, websocket, messages: list, model: str, temperature: float, max_tokens: int, task_id: str, session_id: int) -> str:
        """Stream local model response with dual response format"""
        await self.initialize_model(model)
        
        # Build message history
        chat_messages = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                chat_messages.append({"role": "system", "content": content})
            elif role == 'user':
                chat_messages.append({"role": "user", "content": content})
            elif role == 'assistant':
                chat_messages.append({"role": "assistant", "content": content})
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=max(0.3, min(temperature, 0.8)),
            max_tokens=min(max_tokens, 500),
            n=1,
            presence_penalty=0.2,
            ignore_eos=False
        )
        
        accumulated_response = ""
        
        try:
            # Use chat_stream if available
            if hasattr(self.llm, 'chat_stream'):
                import inspect
                sig = inspect.signature(self.llm.chat_stream)
                if 'sampling_params' in sig.parameters:
                    stream_iterator = self.llm.chat_stream(chat_messages, sampling_params=sampling_params)
                else:
                    stream_iterator = self.llm.chat_stream(chat_messages)
            else:
                prompt = self.llm._messages_to_prompt(chat_messages)
                stream_iterator = self.llm.stream(prompt, sampling_params=sampling_params)
            
            # Handle streaming
            if hasattr(stream_iterator, '__aiter__'):
                async for chunk in stream_iterator:
                    if websocket.close_code is not None:
                        break
                    
                    accumulated_response += chunk
                    
                    # Update database with streaming progress
                    if task_id:
                        self.task_manager.update_dual_response_progress(task_id, 'local', accumulated_response)
                    
                    # Send local model chunk
                    stream_chunk = {
                        "type": "dual_response_chunk",
                        "source": "local",
                        "model": model,
                        "session_id": session_id,
                        "task_id": task_id,
                        "choices": [{
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }]
                    }
                    
                    chunk_json = json.dumps(stream_chunk)
                    logger.debug(f"🔥 LOCAL ASYNC CHUNK: '{chunk}' -> {chunk_json[:200]}...")
                    await websocket.send(chunk_json)
                    await asyncio.sleep(0.02)
            else:
                for chunk in stream_iterator:
                    if websocket.close_code is not None:
                        break
                    
                    accumulated_response += chunk
                    
                    # Update database with streaming progress
                    if task_id:
                        self.task_manager.update_dual_response_progress(task_id, 'local', accumulated_response)
                    
                    # Send local model chunk
                    stream_chunk = {
                        "type": "dual_response_chunk",
                        "source": "local",
                        "model": model,
                        "session_id": session_id,
                        "task_id": task_id,
                        "choices": [{
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }]
                    }
                    
                    chunk_json = json.dumps(stream_chunk)
                    logger.debug(f"🔥 LOCAL SYNC CHUNK: '{chunk}' -> {chunk_json[:200]}...")
                    await websocket.send(chunk_json)
                    await asyncio.sleep(0.02)
            
            # Send local completion
            if websocket.close_code is None:
                # Mark as finished in database
                if task_id:
                    self.task_manager.update_dual_response_progress(task_id, 'local', accumulated_response, finished=True)
                
                local_complete = {
                    "type": "dual_response_chunk",
                    "source": "local",
                    "model": model,
                    "session_id": session_id,
                    "task_id": task_id,
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                await websocket.send(json.dumps(local_complete))
                
                # Note: Don't save to task messages yet - wait for user selection
            
            return accumulated_response
            
        except Exception as e:
            logger.error(f"Error in local streaming: {e}")
            error_chunk = {
                "type": "dual_response_chunk",
                "source": "local",
                "model": model,
                "session_id": session_id,
                "choices": [{
                    "delta": {
                        "content": f"[Local Error: {str(e)}]"
                    },
                    "finish_reason": "stop"
                }]
            }
            await websocket.send(json.dumps(error_chunk))
            return f"[Local Error: {str(e)}]"
    
    async def _stream_gpt_response(self, websocket, messages: list, temperature: float, max_tokens: int, session_id: int, task_id: str = None) -> str:
        """Stream GPT-4.1 response with dual response format"""
        if not self.openai_client:
            error_chunk = {
                "type": "dual_response_chunk",
                "source": "gpt",
                "model": "gpt-4.1",
                "session_id": session_id,
                "choices": [{
                    "delta": {
                        "content": "[GPT Error: OpenAI API key not configured]"
                    },
                    "finish_reason": "stop"
                }]
            }
            await websocket.send(json.dumps(error_chunk))
            return "[GPT Error: OpenAI API key not configured]"
        
        try:
            # Convert messages to OpenAI format
            openai_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role in ['system', 'user', 'assistant']:
                    openai_messages.append({
                        "role": role,
                        "content": content
                    })
            
            # Create streaming response
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            accumulated_response = ""
            async for chunk in response:
                if websocket.close_code is not None:
                    break
                
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    accumulated_response += content
                    
                    # Update database with streaming progress
                    if task_id:
                        self.task_manager.update_dual_response_progress(task_id, 'gpt', accumulated_response)
                    
                    # Send GPT chunk
                    stream_chunk = {
                        "type": "dual_response_chunk",
                        "source": "gpt",
                        "model": "gpt-4.1",
                        "session_id": session_id,
                        "task_id": task_id,
                        "choices": [{
                            "delta": {
                                "content": content
                            },
                            "finish_reason": None
                        }]
                    }
                    
                    chunk_json = json.dumps(stream_chunk)
                    logger.debug(f"🔥 GPT CHUNK: '{content}' -> {chunk_json[:200]}...")
                    await websocket.send(chunk_json)
                    await asyncio.sleep(0.02)
            
            # Send GPT completion
            if websocket.close_code is None:
                # Mark as finished in database
                if task_id:
                    self.task_manager.update_dual_response_progress(task_id, 'gpt', accumulated_response, finished=True)
                
                gpt_complete = {
                    "type": "dual_response_chunk",
                    "source": "gpt",
                    "model": "gpt-4.1",
                    "session_id": session_id,
                    "task_id": task_id,
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                await websocket.send(json.dumps(gpt_complete))
            
            return accumulated_response
            
        except Exception as e:
            logger.error(f"Error streaming GPT response: {e}")
            error_chunk = {
                "type": "dual_response_chunk",
                "source": "gpt",
                "model": "gpt-4.1",
                "session_id": session_id,
                "choices": [{
                    "delta": {
                        "content": f"[GPT Error: {str(e)}]"
                    },
                    "finish_reason": "stop"
                }]
            }
            await websocket.send(json.dumps(error_chunk))
            return f"[GPT Error: {str(e)}]"

    async def _stream_local_action_response(self, websocket, messages: list, mentioned_servers: List[str], model: str, temperature: float, max_tokens: int, task_id: str, session_id: int) -> str:
        """Stream local model response with full tool capabilities in dual response format"""
        try:
            console.print(f"🔥 LOCAL_ACTION: Starting local action response streaming")
            console.print(f"🔥   - session_id: {session_id}")
            console.print(f"🔥   - model: {model}")
            console.print(f"🔥   - mentioned_servers: {mentioned_servers}")
            await self.initialize_model(model)
            console.print(f"🔥   - Model initialized successfully")
            console.print(f"🔥   - Model type: {type(self.llm)}")
            console.print(f"🔥   - Model has chat_stream: {hasattr(self.llm, 'chat_stream')}")
            console.print(f"🔥   - Model has stream: {hasattr(self.llm, 'stream')}")
            
            # Build tools schema for mentioned servers
            tools = await self.build_tools_schema(mentioned_servers)
            
            if not tools:
                # No tools found, fall back to chat mode
                logger.warning(f"No tools found for servers: {mentioned_servers}")
                return await self._stream_local_response(websocket, messages, model, temperature, max_tokens, task_id, session_id)
            
            # Build Qwen chat messages with tools
            chat_messages = self.build_qwen_chat_messages(messages, tools)
            
            # Create sampling parameters optimized for tool calling
            sampling_params = SamplingParams(
                temperature=max(0.3, min(temperature, 0.7)),
                max_tokens=min(max_tokens, 600),  # More tokens for thinking + tool calls
                n=1,
                presence_penalty=0.1,  # Reduce presence penalty to prevent over-constraint
                ignore_eos=False
            )
            
            accumulated_response = ""
            
            # Use streaming
            console.print(f"🔥 LOCAL_ACTION: Setting up streaming...")
            console.print(f"🔥   - self.llm type: {type(self.llm)}")
            console.print(f"🔥   - has chat_stream: {hasattr(self.llm, 'chat_stream')}")
            
            if hasattr(self.llm, 'chat_stream'):
                console.print(f"🔥 LOCAL_ACTION: Using chat_stream method")
                import inspect
                sig = inspect.signature(self.llm.chat_stream)
                if 'sampling_params' in sig.parameters:
                    console.print(f"🔥 LOCAL_ACTION: chat_stream accepts sampling_params")
                    console.print(f"🔥 LOCAL_ACTION: About to call chat_stream with sampling_params")
                    console.print(f"🔥   - messages: {len(chat_messages)} messages")
                    console.print(f"🔥   - sampling_params: {sampling_params}")
                    try:
                        stream_iterator = self.llm.chat_stream(chat_messages, sampling_params=sampling_params)
                        console.print(f"🔥 LOCAL_ACTION: chat_stream call successful")
                    except Exception as chat_error:
                        console.print(f"🔥 LOCAL_ACTION: Error in chat_stream call: {chat_error}")
                        console.print(f"🔥 LOCAL_ACTION: Chat error type: {type(chat_error)}")
                        import traceback
                        console.print(f"🔥 LOCAL_ACTION: Chat traceback: {traceback.format_exc()}")
                        raise
                else:
                    console.print(f"🔥 LOCAL_ACTION: chat_stream does NOT accept sampling_params")
                    console.print(f"🔥 LOCAL_ACTION: About to call chat_stream without sampling_params")
                    console.print(f"🔥   - messages: {len(chat_messages)} messages")
                    try:
                        stream_iterator = self.llm.chat_stream(chat_messages)
                        console.print(f"🔥 LOCAL_ACTION: chat_stream call successful")
                    except Exception as chat_error:
                        console.print(f"🔥 LOCAL_ACTION: Error in chat_stream call: {chat_error}")
                        console.print(f"🔥 LOCAL_ACTION: Chat error type: {type(chat_error)}")
                        import traceback
                        console.print(f"🔥 LOCAL_ACTION: Chat traceback: {traceback.format_exc()}")
                        raise
            else:
                console.print(f"🔥 LOCAL_ACTION: Using fallback _messages_to_prompt method")
                prompt = self.llm._messages_to_prompt(chat_messages)
                stream_iterator = self.llm.stream(prompt, sampling_params=sampling_params)
            
            console.print(f"🔥 LOCAL_ACTION: Stream iterator created: {type(stream_iterator)}")
            
            # Handle streaming
            console.print(f"🔥 LOCAL_ACTION: About to start streaming...")
            console.print(f"🔥   - stream_iterator type: {type(stream_iterator)}")
            console.print(f"🔥   - has __aiter__: {hasattr(stream_iterator, '__aiter__')}")
            console.print(f"🔥   - stream_iterator object: {stream_iterator}")
            
            if hasattr(stream_iterator, '__aiter__'):
                console.print(f"🔥 LOCAL_ACTION: Using async iterator")
                chunk_count = 0
                
                # Add timeout to prevent hanging
                async def stream_with_timeout():
                    nonlocal accumulated_response, chunk_count
                    console.print(f"🔥 LOCAL_ACTION: About to enter async for loop with stream_iterator")
                    console.print(f"🔥 LOCAL_ACTION: stream_iterator object: {stream_iterator}")
                    console.print(f"🔥 LOCAL_ACTION: stream_iterator type: {type(stream_iterator)}")
                    console.print(f"🔥 LOCAL_ACTION: stream_iterator dir: {dir(stream_iterator)}")
                    
                    try:
                        async for chunk in stream_iterator:
                            chunk_count += 1
                            if chunk_count <= 5:  # Log first 5 chunks
                                console.print(f"🔥 LOCAL_ACTION: Chunk {chunk_count}: '{chunk}' ({len(chunk)} chars)")
                            elif chunk_count == 6:
                                console.print(f"🔥 LOCAL_ACTION: ... continuing streaming (will only log important events)")
                            
                            if websocket.close_code is not None:
                                console.print(f"🔥 LOCAL_ACTION: WebSocket closed, breaking")
                                break
                            
                            accumulated_response += chunk
                            
                            # Send chunk without tool execution in dual mode
                            stream_chunk = {
                                "type": "dual_response_chunk",
                                "source": "local",
                                "model": f"{model} (with tools)",
                                "session_id": session_id,
                                "choices": [{
                                    "delta": {
                                        "content": chunk
                                    },
                                    "finish_reason": None
                                }]
                            }
                            
                            await websocket.send(json.dumps(stream_chunk))
                            await asyncio.sleep(0.02)
                        
                        console.print(f"🔥 LOCAL_ACTION: Async streaming completed with {chunk_count} chunks")
                        
                    except Exception as loop_error:
                        console.print(f"🔥 LOCAL_ACTION: Error in async for loop: {loop_error}")
                        console.print(f"🔥 LOCAL_ACTION: Loop error type: {type(loop_error)}")
                        import traceback
                        console.print(f"🔥 LOCAL_ACTION: Loop traceback: {traceback.format_exc()}")
                        raise
                
                try:
                    await asyncio.wait_for(stream_with_timeout(), timeout=30.0)
                except asyncio.TimeoutError:
                    console.print(f"🔥 LOCAL_ACTION: Streaming timeout after 30 seconds")
                    logger.error(f"Local streaming timeout for session {session_id}")
                    if not accumulated_response:
                        accumulated_response = "[Local Error: Streaming timeout - model took too long to respond]"
            else:
                for chunk in stream_iterator:
                    if websocket.close_code is not None:
                        break
                    
                    accumulated_response += chunk
                    
                    # Send chunk without tool execution in dual mode
                    stream_chunk = {
                        "type": "dual_response_chunk",
                        "source": "local",
                        "model": f"{model} (with tools)",
                        "session_id": session_id,
                        "choices": [{
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }]
                    }
                    
                    await websocket.send(json.dumps(stream_chunk))
                    await asyncio.sleep(0.02)
            
            # Send completion - NOTE: Don't save to task in dual mode, wait for user selection
            if websocket.close_code is None:
                local_complete = {
                    "type": "dual_response_chunk",
                    "source": "local",
                    "model": f"{model} (with tools)",
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                await websocket.send(json.dumps(local_complete))
            
            return accumulated_response
            
        except Exception as e:
            logger.error(f"Error in local action streaming: {e}")
            error_chunk = {
                "type": "dual_response_chunk",
                "source": "local",
                "model": f"{model} (with tools)",
                "choices": [{
                    "delta": {
                        "content": f"[Local Error: {str(e)}]"
                    },
                    "finish_reason": "stop"
                }]
            }
            await websocket.send(json.dumps(error_chunk))
            return f"[Local Error: {str(e)}]"


    
    async def _stream_gpt_with_same_prompt(self, websocket, actual_prompt: str, temperature: float, max_tokens: int, session_id: int) -> str:
        """Stream GPT-4.1 response using the exact same prompt as the local model"""
        console.print(f"🔥 GPT_ACTION: Starting GPT action response streaming")
        console.print(f"🔥   - session_id: {session_id}")
        console.print(f"🔥   - actual_prompt length: {len(actual_prompt)}")
        console.print(f"🔥   - actual_prompt preview: {actual_prompt[:200]}...")
        
        if not self.openai_client:
            console.print(f"🔥 GPT_ACTION: ERROR - No OpenAI client configured")
            error_chunk = {
                "type": "dual_response_chunk",
                "source": "gpt",
                "model": "gpt-4.1 (reasoning only)",
                "session_id": session_id,
                "choices": [{
                    "delta": {
                        "content": "[GPT Error: OpenAI API key not configured]"
                    },
                    "finish_reason": "stop"
                }]
            }
            await websocket.send(json.dumps(error_chunk))
            return "[GPT Error: OpenAI API key not configured]"
        
        try:
            console.print(f"🔥 GPT_ACTION: OpenAI client available, starting processing...")
            # Convert the Qwen prompt format to OpenAI messages format
            # Parse the actual_prompt to extract the conversation
            import re
            
            openai_messages = []
            
            # Split by message markers
            parts = re.split(r'<\|im_start\|>(system|user|assistant)\n', actual_prompt)
            
            current_role = None
            for i, part in enumerate(parts):
                if part.strip() in ['system', 'user', 'assistant']:
                    current_role = part.strip()
                elif current_role and part.strip():
                    # Clean up the content (remove end markers)
                    content = re.sub(r'<\|im_end\|>.*$', '', part, flags=re.DOTALL).strip()
                    if content:
                        openai_messages.append({
                            "role": current_role,
                            "content": content
                        })
                    current_role = None
            
            # If parsing failed, create a simple prompt-based message
            if not openai_messages:
                openai_messages = [{
                    "role": "user",
                    "content": f"Continue this conversation with the exact same style and format:\n\n{actual_prompt}"
                }]
            
            # Create streaming response
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1",  # Use gpt-4.1 for larger context window (1M tokens vs 8K for gpt-4)
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            accumulated_response = ""
            console.print(f"🔥 GPT_ACTION: About to start streaming from OpenAI...")
            chunk_count = 0
            
            # Add timeout to prevent hanging for GPT streaming too
            async def gpt_stream_with_timeout():
                nonlocal accumulated_response, chunk_count
                async for chunk in response:
                    chunk_count += 1
                    if chunk_count <= 5:  # Log first 5 chunks
                        console.print(f"🔥 GPT_ACTION: Chunk {chunk_count}: {chunk}")
                    elif chunk_count == 6:
                        console.print(f"🔥 GPT_ACTION: ... continuing streaming (will only log important events)")
                    
                    if websocket.close_code is not None:
                        console.print(f"🔥 GPT_ACTION: WebSocket closed, breaking")
                        break
                    
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        accumulated_response += content
                        
                        if chunk_count <= 5:  # Log first 5 chunks
                            console.print(f"🔥 GPT_ACTION: Sending chunk {chunk_count}: '{content}' ({len(content)} chars)")
                        
                        # Send GPT chunk
                        stream_chunk = {
                            "type": "dual_response_chunk",
                            "source": "gpt",
                            "model": "gpt-4.1 (reasoning only)",
                            "session_id": session_id,
                            "choices": [{
                                "delta": {
                                    "content": content
                                },
                                "finish_reason": None
                            }]
                        }
                        
                        await websocket.send(json.dumps(stream_chunk))
                        await asyncio.sleep(0.02)
                console.print(f"🔥 GPT_ACTION: Streaming completed with {chunk_count} chunks")
            
            try:
                await asyncio.wait_for(gpt_stream_with_timeout(), timeout=60.0)
            except asyncio.TimeoutError:
                console.print(f"🔥 GPT_ACTION: Streaming timeout after 60 seconds")
                logger.error(f"GPT streaming timeout for session {session_id}")
                if not accumulated_response:
                    accumulated_response = "[GPT Error: Streaming timeout - model took too long to respond]"
            
            # Send GPT completion
            if websocket.close_code is None:
                gpt_complete = {
                    "type": "dual_response_chunk",
                    "source": "gpt",
                    "model": "gpt-4.1 (reasoning only)",
                    "session_id": session_id,
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                await websocket.send(json.dumps(gpt_complete))
            
            return accumulated_response
            
        except Exception as e:
            logger.error(f"Error streaming GPT response: {e}")
            error_chunk = {
                "type": "dual_response_chunk",
                "source": "gpt",
                "model": "gpt-4.1 (reasoning only)",
                "session_id": session_id,
                "choices": [{
                    "delta": {
                        "content": f"[GPT Error: {str(e)}]"
                    },
                    "finish_reason": "stop"
                }]
            }
            await websocket.send(json.dumps(error_chunk))
            return f"[GPT Error: {str(e)}]"
    
    async def initialize_model(self, model_key: str = "qwen3"):
        """Initialize the AsyncOnlineLLM with the specified model"""
        if self.llm is not None and self.current_model == model_key:
            return  # Model already loaded
        
        model_name = self.models.get(model_key, self.models["qwen3"])
        
        logger.info(f"Initializing model: {model_name}")
        
        # Configure batch processing
        batch_config = BatchConfig(
            max_batch_size=4,
            auto_batch_size=True,
            memory_threshold=0.8,
            min_batch_size=1
        )
        
        try:
            self.llm = AsyncOnlineLLM(
                model=model_name,
                temperature=0.7,
                max_tokens=2048,
                engine="hf",  # Use HuggingFace engine
                lora_rank=16,
                lora_alpha=64,
                batch_config=batch_config,
                train_every=3,
                use_adam8bit=True,
                thinking_mode=False,
            )
            self.current_model = model_key
            logger.info(f"Model {model_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            raise



    def messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI messages format to a single prompt string"""
        prompt_parts = []
        
        # Add system instruction to prevent conversational hallucinations
        prompt_parts.append("You are a helpful assistant. Answer questions directly and concisely. Do not generate fake conversations or roleplay scenarios.")
        prompt_parts.append("")  # Empty line for separation
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def get_mcp_servers(self, query: str = "") -> list:
        """Get list of MCP servers, optionally filtered by query"""
        servers = []
        
        for server in self.mcp_servers:
            # Check if all required environment variables are available
            required_env_vars = server.get("required_env", [])
            api_key_available = all(get_api_key(env_var) for env_var in required_env_vars) if required_env_vars else True
            
            server_info = {
                "id": server["id"],
                "name": server["name"],
                "description": server["description"],
                "required_env": server.get("required_env", []),
                "optional_env": server.get("optional_env", []),
                "api_key_available": api_key_available
            }
            
            # Filter by query if provided
            if query:
                query_lower = query.lower()
                if not (query_lower in server["name"].lower() or 
                       query_lower in server["description"].lower() or
                       query_lower in server["id"].lower()):
                    continue
            
            servers.append(server_info)
        
        return servers

    async def get_mcp_client(self, server_id: str):
        """Get or create MCP client for a server"""
        if server_id in self.mcp_clients:
            return self.mcp_clients[server_id]
        
        # Find server config
        server_config = None
        for server in self.mcp_servers:
            if server["id"] == server_id:
                server_config = server
                break
        
        if not server_config:
            raise ValueError(f"Server {server_id} not found")
        
        # Check for required environment variables
        for env_var in server_config.get("required_env", []):
            if not get_api_key(env_var):
                raise ValueError(f"Required environment variable {env_var} not configured for server {server_id}")
        
        # Create MCP config with dynamic environment variable population
        config = server_config["config"].copy()
        
        if "env" in config:
            # Get all required and optional env vars
            all_env_vars = server_config.get("required_env", []) + server_config.get("optional_env", [])
            for env_var in all_env_vars:
                if env_var in config["env"]:
                    config["env"][env_var] = get_api_key(env_var)
        
        mcp_config = {
            "mcpServers": {
                server_id: config
            }
        }
        
        # Create client
        client = MCPClient.from_dict(mcp_config)
        self.mcp_clients[server_id] = client
        
        return client

    async def get_mcp_session(self, server_id: str):
        """Get or create MCP session for a server"""
        if server_id in self.mcp_sessions:
            return self.mcp_sessions[server_id]
        
        client = await self.get_mcp_client(server_id)
        session = await client.create_session(server_id)
        
        # Connect if not already connected
        if not session.connector.is_connected:
            await session.connect()
        
        self.mcp_sessions[server_id] = session
        return session

    async def cleanup_mcp_connections(self):
        """Clean up MCP connections and clear caches"""
        for client in self.mcp_clients.values():
            try:
                await client.close_all_sessions()
            except Exception as e:
                logger.error(f"Error closing MCP client: {e}")
        
        self.mcp_clients.clear()
        self.mcp_sessions.clear()
        # Clear actions cache when connections are cleaned up
        cache_count = len(self.mcp_actions_cache)
        self.mcp_actions_cache.clear()
        logger.info(f"🧹 Cleared MCP connections and actions cache ({cache_count} servers)")

    def clear_mcp_actions_cache(self, server_id: str = None):
        """Clear MCP actions cache for a specific server or all servers"""
        if server_id:
            if server_id in self.mcp_actions_cache:
                del self.mcp_actions_cache[server_id]
                logger.info(f"🗑️ Cleared actions cache for {server_id}")
        else:
            cache_count = len(self.mcp_actions_cache)
            self.mcp_actions_cache.clear()
            logger.info(f"🗑️ Cleared all MCP actions cache ({cache_count} servers)")

    async def refresh_mcp_actions(self, server_id: str) -> dict:
        """Refresh actions for a specific server by clearing cache and re-fetching"""
        self.clear_mcp_actions_cache(server_id)
        return await self.get_mcp_server_actions([server_id])

    def get_mcp_cache_stats(self) -> dict:
        """Get statistics about the MCP actions cache"""
        return {
            "cached_servers": list(self.mcp_actions_cache.keys()),
            "cache_counts": {
                server_id: len(actions) for server_id, actions in self.mcp_actions_cache.items()
            },
            "total_cached_servers": len(self.mcp_actions_cache),
            "total_cached_actions": sum(len(actions) for actions in self.mcp_actions_cache.values())
        }

    async def get_mcp_server_actions(self, server_ids: list) -> dict:
        """Get available actions for specified MCP servers with caching"""
        server_actions = {}
        
        for server_id in server_ids:
            # Check cache first
            if server_id in self.mcp_actions_cache:
                server_actions[server_id] = self.mcp_actions_cache[server_id]
                logger.info(f"✓ Using cached actions for {server_id}: {len(server_actions[server_id])} actions")
                continue
            
            try:
                # Get real tools from MCP server
                session = await self.get_mcp_session(server_id)
                tools = await session.connector.list_tools()
                
                # Convert MCP tools to actions format
                actions = []
                for tool in tools:
                    # Build parameter information from inputSchema in JSON schema format
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                    
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        schema = tool.inputSchema
                        if isinstance(schema, dict) and 'properties' in schema:
                            required = schema.get('required', [])
                            for param_name, param_info in schema['properties'].items():
                                param_type = param_info.get('type', 'string')
                                param_desc = param_info.get('description', '')
                                parameters["properties"][param_name] = {
                                    "type": param_type,
                                    "description": param_desc
                                }
                                if param_name in required:
                                    parameters["required"].append(param_name)
                    
                    action = {
                        "name": tool.name,
                        "description": tool.description or "No description available",
                        "parameters": parameters
                    }
                    actions.append(action)
                
                # Cache the actions
                self.mcp_actions_cache[server_id] = actions
                server_actions[server_id] = actions
                logger.info(f"📥 Fetched and cached {len(actions)} actions for {server_id}")
                
            except Exception as e:
                logger.error(f"Error getting actions for server {server_id}: {e}")
                # Return error information (don't cache errors)
                server_actions[server_id] = [{
                    "name": "error",
                    "description": f"Failed to connect to {server_id}: {str(e)}",
                    "parameters": {}
                }]
        
        return server_actions

    async def build_mcp_tool_descriptions(self, server_ids: List[str]) -> List[str]:
        """Build detailed tool descriptions following train_one_vl.py patterns"""
        tool_descriptions = []
        
        # Get actions for all requested servers
        server_actions = await self.get_mcp_server_actions(server_ids)
        
        for server_id, actions in server_actions.items():
            for action in actions:
                if action.get("name") == "error":
                    continue  # Skip error actions
                
                # Build tool description like in train_one_vl.py
                tool_name = action.get("name", "unknown")
                description = action.get("description", "No description")
                
                tool_desc = f"{tool_name}: {description}"
                
                # Add parameter information from action parameters
                parameters = action.get("parameters", {})
                if parameters:
                    params = []
                    for param_name, param_info in parameters.items():
                        param_type = param_info.get("type", "unknown")
                        is_required = param_info.get("required", False)
                        required_mark = '*' if is_required else ''
                        params.append(f"{required_mark}{param_name}({param_type})")
                    
                    if params:
                        tool_desc += f" - Parameters: {', '.join(params)}"
                
                tool_descriptions.append(tool_desc)
        
        return tool_descriptions

    async def build_tools_schema(self, server_ids: List[str]) -> List[dict]:
        """Build tools schema in Qwen format"""
        tools = []
        
        # Get actions for all requested servers
        server_actions = await self.get_mcp_server_actions(server_ids)
        
        for server_id, actions in server_actions.items():
            for action in actions:
                if action.get("name") == "error":
                    continue  # Skip error actions
                
                tool_name = action.get("name", "unknown")
                description = action.get("description", "No description")
                parameters = action.get("parameters", {})
                
                # Convert to Qwen tools format
                tool_schema = {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters  # Parameters are already in the correct JSON schema format
                }
                
                tools.append(tool_schema)
        
        return tools

    def build_qwen_chat_messages(self, messages: List[dict], tools: List[dict] = None) -> List[dict]:
        """Build messages in Qwen chat template format with tools"""
        chat_messages = []
        
        # Build system message with tools
        system_content = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        
        if tools:
            system_content += "\n\n" + "❗"*60 + "\n"
            system_content += "THIS IS NOT A CODING TUTORIAL - YOU HAVE REAL TOOLS!\n"
            system_content += "❗"*60 + "\n\n"
            
            system_content += "STOP generating JavaScript examples!\n"
            system_content += "STOP saying 'you need to write a script'!\n"
            system_content += "STOP explaining how to use Playwright!\n\n"
            
            system_content += "YOU ARE A ROBOT WITH BROWSER CONTROL.\n"
            system_content += "When asked to find restaurants, YOU go find them.\n"
            system_content += "When asked to navigate, YOU navigate.\n"
            system_content += "When asked to click, YOU click.\n\n"
            
            system_content += "IMMEDIATE ACTION REQUIRED:\n"
            system_content += "- User says 'find restaurants' → YOU call browser_navigate\n"
            system_content += "- User says 'click button' → YOU call browser_click\n"
            system_content += "- User says 'take screenshot' → YOU call browser_take_screenshot\n\n"
            
            system_content += "Available tools (USE THESE NOW!):\n<tools>\n"
            
            for tool in tools:
                system_content += json.dumps(tool, indent=2) + "\n"
            
            system_content += "</tools>\n\n"
            
            system_content += "MANDATORY FORMAT (INCLUDE THE XML TAGS!):\n"
            system_content += "1. First, think about what you need to do:\n"
            system_content += "<think>\nI need to [explain your reasoning here]\n</think>\n\n"
            system_content += "2. Then make the tool call:\n"
            system_content += "<tool_call>\n"
            system_content += '{"name": "actual_tool_name", "arguments": {"actual_param": "actual_value"}}\n'
            system_content += "</tool_call>\n\n"
            
            system_content += "CRITICAL: You MUST include both <think> and <tool_call> XML tags!\n"
            system_content += "Example response:\n"
            system_content += "<think>\nI need to navigate to a restaurant website to find good restaurants in Boulder.\n</think>\n\n"
            system_content += "<tool_call>\n"
            system_content += '{"name": "browser_navigate", "arguments": {"url": "https://yelp.com"}}\n'
            system_content += "</tool_call>\n\n"
            
            system_content += "RIGHT NOW: If the user mentions restaurants, immediately navigate to a restaurant site!\n"
            system_content += "RIGHT NOW: If the user mentions websites, immediately navigate there!\n"
            system_content += "NO EXPLANATIONS. NO CODE. JUST TOOL CALLS!"
        
        chat_messages.append({"role": "system", "content": system_content})
        
        # Add conversation messages
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role in ['user', 'assistant', 'system']:
                chat_messages.append({"role": role, "content": content})
        
        # Log the complete prompt being sent to model
        self._log_prompt_to_model(chat_messages, tools)
        
        return chat_messages
    
    def _log_final_processed_prompt(self, final_prompt: str):
        """Log the final processed prompt that goes to the LLM with all formatting"""
        console.print("\n" + "🔥"*100)
        console.print(Panel(
            "🚀 FINAL PROCESSED PROMPT TO LLM",
            style="bold red",
            border_style="red"
        ))
        
        console.print(Panel(
            final_prompt,
            title=f"📤 EXACT PROMPT SENT TO MODEL ({len(final_prompt)} chars)",
            style="white",
            border_style="red"
        ))
        
        console.print("🔥"*100 + "\n")
    
    def _log_prompt_to_model(self, chat_messages: List[dict], tools: List[dict] = None):
        """Log the prompt being sent to the model with Rich formatting"""
        console.print("\n" + "="*100)
        console.print(Panel(
            "🤖 SENDING PROMPT TO MODEL",
            style="bold blue",
            border_style="blue"
        ))
        
        # Show tools if present
        if tools:
            tools_table = Table(title="🛠️ Available Tools", box=box.MINIMAL)
            tools_table.add_column("Name", style="cyan", no_wrap=True)
            tools_table.add_column("Description", style="white")
            
            for tool in tools:
                tools_table.add_row(
                    tool.get("name", "unknown"),
                    tool.get("description", "No description")[:80] + "..." if len(tool.get("description", "")) > 80 else tool.get("description", "")
                )
            
            console.print(tools_table)
        
        # Show each message with full content
        for i, msg in enumerate(chat_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Color code by role
            if role == "system":
                style = "yellow"
                icon = "⚙️"
            elif role == "user":
                style = "green"
                icon = "👤"
            elif role == "assistant":
                style = "blue"
                icon = "🤖"
            else:
                style = "white"
                icon = "❓"
            
            # Show FULL content for debugging - no truncation
            console.print(Panel(
                content,
                title=f"{icon} {role.upper()} MESSAGE {i+1} ({len(content)} chars)",
                style=style,
                border_style=style
            ))
        
        console.print("="*100 + "\n")

    def _log_model_response(self, response: str, response_type: str = "STREAMING"):
        """Log the model's response with Rich formatting"""
        console.print("\n" + "="*100)
        console.print(Panel(
            f"🤖 MODEL {response_type} RESPONSE",
            style="bold magenta",
            border_style="magenta"
        ))
        
        # Show FULL response - no truncation
        console.print(Panel(
            response,
            title=f"📝 RAW MODEL OUTPUT ({len(response)} chars)",
            style="white",
            border_style="magenta"
        ))
        
        # Check for tool calls and thinking
        if "<tool_call>" in response:
            console.print("🔧 **TOOL CALL DETECTED** - Model is requesting tool execution")
        if "<think>" in response:
            console.print("🧠 **THINKING DETECTED** - Model is showing reasoning")
        
        console.print("="*100 + "\n")

    def extract_action_from_response(self, response: str) -> dict:
        """Extract action from response with thinking support (Qwen format)"""
        result = {}
        
        # Extract thinking if present
        if "<think>" in response and "</think>" in response:
            think_start = response.find("<think>") + 7
            think_end = response.find("</think>")
            thinking = response[think_start:think_end].strip()
            result["thinking"] = thinking
        
        # Extract tool call in Qwen format
        if "<tool_call>" in response and "</tool_call>" in response:
            tool_start = response.find("<tool_call>") + 11
            tool_end = response.find("</tool_call>")
            tool_str = response[tool_start:tool_end].strip()
            
            try:
                tool_call = json.loads(tool_str)
                
                # Convert from Qwen format {"name": "...", "arguments": {...}} 
                # to our internal format {"tool": "...", "parameters": {...}}
                if "name" in tool_call:
                    result["tool"] = tool_call["name"]
                    result["parameters"] = tool_call.get("arguments", {})
                    result["original_format"] = tool_call  # Keep original for reference
                    return result
                else:
                    return {"error": "Tool call missing 'name' field", **result}
                    
            except Exception as e:
                logger.error(f"Error parsing tool call JSON: {e}")
                return {"error": f"Invalid JSON: {str(e)}", **result}
        
        return {"error": "No tool_call tags found", **result}

    def extract_mentioned_servers(self, text: str) -> List[str]:
        """Extract @server mentions from text"""
        # Find all @server_id mentions
        server_mentions = re.findall(r'@([a-zA-Z0-9_-]+)', text)
        
        # Filter to only include actual server IDs we know about
        known_server_ids = {server["id"] for server in self.mcp_servers}
        mentioned_servers = [server_id for server_id in server_mentions if server_id in known_server_ids]
        
        return list(set(mentioned_servers))  # Remove duplicates

    async def should_use_action_mode(self, messages: List[dict], task_id: str = None) -> tuple[bool, List[str]]:
        """Determine if we should use action mode based on MCP server mentions or task context"""
        # First check if task has associated MCP servers
        if task_id:
            task_data = self.get_task(task_id)
            if not task_data.get('error'):
                task_servers = task_data.get('task', {}).get('mcp_servers', [])
                if task_servers:
                    logger.info(f"🎯 Task {task_id} has MCP servers: {task_servers} - using action mode")
                    return True, task_servers
        
        # Check the latest user message for @server mentions
        if not messages:
            return False, []
        
        latest_message = messages[-1]
        if latest_message.get("role") != "user":
            return False, []
        
        content = latest_message.get("content", "")
        mentioned_servers = self.extract_mentioned_servers(content)
        
        # Use action mode if servers are mentioned
        return len(mentioned_servers) > 0, mentioned_servers

    def _serialize_mcp_result(self, result, depth=0):
        """Safely serialize MCP results to JSON-compatible format"""
        indent = "  " * depth
        try:
            logger.debug(f"{indent}🔍 Serializing object: type={type(result).__name__}, value={str(result)[:100]}...")
            
            # Handle different types of MCP results
            if hasattr(result, 'model_dump'):
                logger.debug(f"{indent}✅ Using model_dump() for {type(result).__name__}")
                # Pydantic model - use model_dump
                serialized = result.model_dump()
                logger.debug(f"{indent}✅ model_dump() successful: {str(serialized)[:200]}...")
                return serialized
            elif hasattr(result, '__dict__'):
                logger.debug(f"{indent}🔧 Converting object with __dict__ for {type(result).__name__}")
                # Object with attributes - convert to dict
                obj_dict = {}
                for key, value in result.__dict__.items():
                    logger.debug(f"{indent}  🔑 Processing attribute: {key} = {type(value).__name__}")
                    if isinstance(value, (str, int, float, bool, type(None))):
                        obj_dict[key] = value
                        logger.debug(f"{indent}  ✅ Simple type: {key} = {value}")
                    elif isinstance(value, list):
                        logger.debug(f"{indent}  📋 List with {len(value)} items")
                        obj_dict[key] = [self._serialize_mcp_result(item, depth + 1) for item in value]
                        logger.debug(f"{indent}  ✅ List serialized: {key}")
                    elif hasattr(value, 'model_dump'):
                        logger.debug(f"{indent}  🎯 Using model_dump for nested object: {type(value).__name__}")
                        obj_dict[key] = value.model_dump()
                        logger.debug(f"{indent}  ✅ Nested model_dump successful: {key}")
                    elif hasattr(value, '__dict__'):
                        logger.debug(f"{indent}  🔄 Recursing into nested object: {type(value).__name__}")
                        obj_dict[key] = self._serialize_mcp_result(value, depth + 1)
                        logger.debug(f"{indent}  ✅ Nested object serialized: {key}")
                    else:
                        logger.debug(f"{indent}  🔤 Converting to string: {type(value).__name__}")
                        obj_dict[key] = str(value)
                        logger.debug(f"{indent}  ✅ String conversion: {key} = {str(value)[:50]}...")
                logger.debug(f"{indent}✅ Object dict serialization complete")
                return obj_dict
            elif isinstance(result, list):
                logger.debug(f"{indent}📋 Serializing list with {len(result)} items")
                # List - serialize each item
                serialized_list = [self._serialize_mcp_result(item, depth + 1) for item in result]
                logger.debug(f"{indent}✅ List serialization complete")
                return serialized_list
            elif isinstance(result, dict):
                logger.debug(f"{indent}📖 Serializing dict with {len(result)} keys")
                # Dict - serialize each value
                serialized_dict = {key: self._serialize_mcp_result(value, depth + 1) for key, value in result.items()}
                logger.debug(f"{indent}✅ Dict serialization complete")
                return serialized_dict
            else:
                logger.debug(f"{indent}🔤 Fallback to string for {type(result).__name__}")
                # Fallback to string representation
                str_result = str(result)
                logger.debug(f"{indent}✅ String fallback: {str_result[:100]}...")
                return str_result
        except Exception as e:
            logger.error(f"{indent}❌ Error serializing MCP result at depth {depth}: {e}")
            logger.error(f"{indent}❌ Object type: {type(result).__name__}")
            logger.error(f"{indent}❌ Object attributes: {dir(result) if hasattr(result, '__dict__') else 'No __dict__'}")
            import traceback
            logger.error(f"{indent}❌ Full traceback: {traceback.format_exc()}")
            return str(result)

    async def execute_mcp_action(self, server_id: str, action_name: str, parameters: dict) -> dict:
        """Execute an MCP action on a server"""
        try:
            logger.info(f"🚀 Executing MCP action: {action_name} on server {server_id}")
            logger.debug(f"📋 Parameters: {json.dumps(parameters, indent=2)}")
            
            session = await self.get_mcp_session(server_id)
            logger.debug(f"✅ Got MCP session for {server_id}")
            
            result = await session.connector.call_tool(action_name, parameters)
            logger.info(f"✅ MCP action executed successfully: {action_name}")
            logger.debug(f"🔍 Raw result type: {type(result).__name__}")
            logger.debug(f"🔍 Raw result attributes: {dir(result) if hasattr(result, '__dict__') else 'No __dict__'}")
            logger.debug(f"🔍 Raw result string representation: {str(result)[:200]}...")
            
            # Test JSON serialization on the raw result to see where it fails
            try:
                json.dumps(result)
                logger.debug("✅ Raw result is JSON serializable")
            except Exception as json_error:
                logger.warning(f"❌ Raw result is NOT JSON serializable: {json_error}")
                logger.debug(f"❌ JSON error type: {type(json_error).__name__}")
            
            # Safely serialize the result
            logger.debug("🔄 Starting serialization process...")
            serialized_result = self._serialize_mcp_result(result)
            logger.info("✅ Serialization completed successfully")
            
            # Test JSON serialization on the serialized result
            try:
                json.dumps(serialized_result)
                logger.debug("✅ Serialized result is JSON serializable")
            except Exception as json_error:
                logger.error(f"❌ Serialized result is STILL NOT JSON serializable: {json_error}")
                logger.error(f"❌ Serialized result type: {type(serialized_result).__name__}")
                logger.error(f"❌ Serialized result: {str(serialized_result)[:500]}...")
                # Fallback to string representation
                serialized_result = str(result)
            
            # Extract content if available
            logger.debug("🔄 Extracting content...")
            content = ""
            if hasattr(result, 'content'):
                logger.debug(f"📝 Found content attribute: type={type(result.content).__name__}")
                if isinstance(result.content, list):
                    logger.debug(f"📋 Content is a list with {len(result.content)} items")
                    # Handle list of content items (common in MCP)
                    content_parts = []
                    for i, item in enumerate(result.content):
                        logger.debug(f"  🔍 Content item {i}: type={type(item).__name__}")
                        if hasattr(item, 'text'):
                            logger.debug(f"  📝 Found text attribute in item {i}")
                            content_parts.append(item.text)
                        elif isinstance(item, str):
                            logger.debug(f"  🔤 Item {i} is a string")
                            content_parts.append(item)
                        else:
                            logger.debug(f"  🔄 Converting item {i} to string")
                            content_parts.append(str(item))
                    content = "\n".join(content_parts)
                    logger.debug(f"✅ Extracted content from list: {len(content)} characters")
                else:
                    logger.debug("🔤 Content is not a list, converting to string")
                    content = str(result.content)
                    logger.debug(f"✅ Extracted content: {len(content)} characters")
            else:
                logger.debug("❌ No content attribute found, using string representation")
                content = str(result)
                logger.debug(f"✅ Using result string: {len(content)} characters")
            
            final_result = {
                "success": True,
                "result": serialized_result,
                "content": content
            }
            
            # Final JSON serialization test
            try:
                json.dumps(final_result)
                logger.info("✅ Final result is JSON serializable")
            except Exception as json_error:
                logger.error(f"❌ Final result is NOT JSON serializable: {json_error}")
                # Ultimate fallback
                final_result = {
                    "success": True,
                    "result": str(result),
                    "content": content
                }
                logger.warning("🔄 Using ultimate fallback with string result")
            
            logger.info(f"🎉 MCP action execution completed: {action_name}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Error executing MCP action {action_name} on server {server_id}: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "content": f"Error executing {action_name}: {str(e)}"
            }

    def create_task(self, initial_prompt: str, model: str = "qwen2.5-vl") -> str:
        """Create a new task and return its ID"""
        return self.task_manager.create_task(initial_prompt, model)
    
    def get_task(self, task_id: str) -> dict:
        """Get task details by ID with current state"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return {"error": "Task not found"}
        
        messages = self.task_manager.get_task_messages(task_id)
        
        # Get current task state (idle, streaming_single, streaming_dual, awaiting_dual_selection)
        current_state = self.task_manager.get_task_state(task_id)
        
        # Get pending dual responses if any
        pending_dual = self.task_manager.get_pending_dual_responses(task_id)
        
        return {
            "task": task,
            "messages": messages,
            "current_state": current_state,
            "pending_dual": pending_dual
        }
    
    def get_recent_tasks(self) -> list:
        """Get list of recent tasks"""
        return self.task_manager.get_recent_tasks()

    async def stream_response(self, websocket, messages: list, model: str = "qwen3", temperature: float = 0.7, max_tokens: int = 2048, task_id: str = None, dual_model: bool = True):
        """Stream response from the actual LLM with MCP action support and optional dual model comparison"""
        
        try:
            # Debug logging for dual model decision
            logger.info(f"🐛 stream_response called with dual_model={dual_model}, task_id={task_id}")
            
            # Initialize model if needed
            await self.initialize_model(model)
            
            # Check if we should use action mode based on MCP server mentions or task context
            use_action_mode, mentioned_servers = await self.should_use_action_mode(messages, task_id)
            logger.info(f"🐛 use_action_mode={use_action_mode}, mentioned_servers={mentioned_servers}")
            
            # Dual mode should always be active when OpenAI client is available
            # Force dual model mode for action mode
            force_dual_for_action = use_action_mode and self.openai_client
            use_dual_mode = force_dual_for_action or (dual_model and self.openai_client)
            
            logger.info(f"🐛 force_dual_for_action={force_dual_for_action}, use_dual_mode={use_dual_mode}")
            logger.info(f"🐛 self.openai_client exists: {self.openai_client is not None}")
            
            if use_dual_mode:
                if force_dual_for_action:
                    logger.info("🔄 Forcing dual model mode for action mode (thinking + tool calls)")
                else:
                    logger.info("🔄 Using dual model mode with GPT-4.1 comparison")
                    
                if use_action_mode:
                    logger.info(f"🎯 Action mode detected! Mentioned servers: {mentioned_servers}")
                    # For action mode, we'll create a dual response version that handles tools
                    await self.stream_dual_action_response(websocket, messages, mentioned_servers, model, temperature, max_tokens, task_id)
                else:
                    logger.info("💬 Standard chat mode")
                    await self.stream_dual_response(websocket, messages, model, temperature, max_tokens, task_id)
            else:
                logger.info("🚫 NOT using dual model mode")
                if dual_model or force_dual_for_action:
                    if force_dual_for_action:
                        logger.warning("⚠️ Action mode requires dual model but OpenAI client not available - falling back to single model")
                    else:
                        logger.warning("⚠️ Dual model requested but OpenAI client not available - falling back to single model")
                
                if use_action_mode:
                    logger.info(f"🎯 Action mode detected! Mentioned servers: {mentioned_servers}")
                    await self.stream_action_response(websocket, messages, mentioned_servers, model, temperature, max_tokens, task_id)
                else:
                    logger.info("💬 Standard chat mode")
                    await self.stream_chat_response(websocket, messages, model, temperature, max_tokens, task_id)
                
        except Exception as e:
            console.print(Panel(
                f"❌ Error during streaming setup: {str(e)}",
                title="🚨 STREAMING ERROR",
                style="bold red",
                border_style="red"
            ))
            import traceback
            traceback.print_exc()
            
            # Send error chunk only if connection is open
            if websocket.close_code is None:
                try:
                    error_chunk = {
                        "choices": [{
                            "delta": {
                                "content": f"\n\n[Error: {str(e)}]"
                            },
                            "finish_reason": "stop"
                        }]
                    }
                    await websocket.send(json.dumps(error_chunk))
                except:
                    pass  # Connection might be closed

    async def stream_chat_response(self, websocket, messages: list, model: str, temperature: float, max_tokens: int, task_id: str = None):
        """Stream standard chat response"""
        # Use chat format instead of prompt conversion to prevent hallucinations
        chat_messages = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                chat_messages.append({"role": "system", "content": content})
            elif role == 'user':
                chat_messages.append({"role": "user", "content": content})
            elif role == 'assistant':
                chat_messages.append({"role": "assistant", "content": content})
        
        # Create sampling parameters with limits to prevent infinite responses
        sampling_params = SamplingParams(
            temperature=max(0.3, min(temperature, 0.8)),  # Keep temperature reasonable
            max_tokens=min(max_tokens, 500),  # Limit response length to prevent runaway generation
            n=1,
            presence_penalty=0.2,  # Increase to reduce repetition
            ignore_eos=False
        )
        
        console.print(Panel(
            f"📨 Streaming chat response for {len(chat_messages)} messages",
            title="💬 CHAT MODE ACTIVATED",
            style="bold magenta",
            border_style="magenta"
        ))
            
        # Stream response from actual LLM using chat format
        accumulated_response = ""
        chunk_count = 0
        
        try:
            # Use chat_stream if available (handles vision inputs and proper Qwen2.5-VL formatting)
            if hasattr(self.llm, 'chat_stream'):
                logger.info("Using chat_stream method")
                # Check if chat_stream accepts sampling_params
                import inspect
                sig = inspect.signature(self.llm.chat_stream)
                if 'sampling_params' in sig.parameters:
                    logger.info("chat_stream accepts sampling_params")
                    stream_iterator = self.llm.chat_stream(chat_messages, sampling_params=sampling_params)
                else:
                    logger.info("chat_stream does not accept sampling_params")
                    stream_iterator = self.llm.chat_stream(chat_messages)
            else:
                logger.info("Fallback: Using _messages_to_prompt method")
                # Fallback: Use OnlineLLM's _messages_to_prompt which handles Qwen2.5-VL formatting
                prompt = self.llm._messages_to_prompt(chat_messages)
                stream_iterator = self.llm.stream(prompt, sampling_params=sampling_params)
            
            # Handle both async generators and regular generators
            if hasattr(stream_iterator, '__aiter__'):
                logger.info("Processing async generator")
                # Async generator
                async for chunk in stream_iterator:
                    # Check if websocket is still open
                    if websocket.close_code is not None:
                        logger.warning("WebSocket closed during streaming")
                        break
                    
                    accumulated_response += chunk
                    chunk_count += 1
                    logger.debug(f"Processed chunk {chunk_count}: '{chunk}'")
                    
                    # Create OpenAI-compatible streaming response
                    stream_chunk = {
                        "choices": [{
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }]
                    }
                    
                    try:
                        await websocket.send(json.dumps(stream_chunk))
                        # Small delay to make streaming visible
                        await asyncio.sleep(0.02)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed while sending chunk")
                        break
                logger.info(f"Async generator completed with {chunk_count} chunks")
            else:
                logger.info("Processing regular generator")
                # Regular generator - convert to async
                for chunk in stream_iterator:
                    # Check if websocket is still open
                    if websocket.close_code is not None:
                        logger.warning("WebSocket closed during streaming")
                        break
                    
                    accumulated_response += chunk
                    chunk_count += 1
                    logger.debug(f"Processed chunk {chunk_count}: '{chunk}'")
                    
                    # Create OpenAI-compatible streaming response
                    stream_chunk = {
                        "choices": [{
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }]
                    }
                    
                    try:
                        await websocket.send(json.dumps(stream_chunk))
                        # Small delay to make streaming visible
                        await asyncio.sleep(0.02)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed while sending chunk")
                        break
                logger.info(f"Regular generator completed with {chunk_count} chunks")
                
            # Send final chunk only if connection is still open
            if websocket.close_code is None:
                final_chunk = {
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                
                await websocket.send(json.dumps(final_chunk))
                logger.info(f"Response completed. Total length: {len(accumulated_response)} characters, {chunk_count} chunks")
                
                # Save assistant response to database if task_id provided
                if task_id and accumulated_response:
                    self.task_manager.add_message(task_id, 'assistant', accumulated_response)
                
                console.print(Panel(
                    f"💾 Saved chat response to task: {task_id}\n"
                    f"📏 Length: {len(accumulated_response)} characters\n"
                    f"📊 Total chunks: {chunk_count}",
                    title="✅ CHAT STREAMING COMPLETED",
                    style="bold green",
                    border_style="green"
                ))
            else:
                console.print(Panel(
                    f"⚠️ Streaming stopped due to closed connection\n"
                    f"📊 Sent {chunk_count} chunks before disconnect",
                    title="🔌 CONNECTION CLOSED",
                    style="bold yellow",
                    border_style="yellow"
                ))
                
        except Exception as stream_error:
            console.print(Panel(
                f"❌ Error in streaming loop: {str(stream_error)}",
                title="🚨 STREAMING ERROR",
                style="bold red",
                border_style="red"
            ))
            if websocket.close_code is None:
                # Try to send error response
                try:
                    error_chunk = {
                        "choices": [{
                            "delta": {
                                "content": f"\n\n[Streaming Error: {str(stream_error)}]"
                            },
                            "finish_reason": "stop"
                        }]
                    }
                    await websocket.send(json.dumps(error_chunk))
                except:
                    pass  # Connection might be closed
            raise stream_error
            
    async def stream_action_response(self, websocket, messages: list, mentioned_servers: List[str], model: str, temperature: float, max_tokens: int, task_id: str = None):
        """Stream action-based response using Qwen chat template with tools"""
        
        try:
            # Build tools schema for mentioned servers
            console.print(f"🔥 LOCAL_ACTION: Building tools schema for servers: {mentioned_servers}")
            tools = await self.build_tools_schema(mentioned_servers)
            console.print(f"🔥 LOCAL_ACTION: Built {len(tools)} tools")
            
            if not tools:
                # No tools found, fall back to chat mode
                logger.warning(f"No tools found for servers: {mentioned_servers}")
                console.print(f"🔥 LOCAL_ACTION: No tools found, falling back to chat mode")
                await self.stream_chat_response(websocket, messages, model, temperature, max_tokens, task_id)
                return
            
            # Build Qwen chat messages with tools
            console.print(f"🔥 LOCAL_ACTION: Building Qwen chat messages with tools...")
            chat_messages = self.build_qwen_chat_messages(messages, tools)
            console.print(f"🔥 LOCAL_ACTION: Built {len(chat_messages)} chat messages")
            
            console.print(Panel(
                f"🛠️ Built Qwen chat with {len(tools)} tools\n"
                f"🎯 Mentioned servers: {', '.join(mentioned_servers)}",
                title="📋 TOOL MODE ACTIVATED",
                style="bold cyan",
                border_style="cyan"
            ))
            
            # Create sampling parameters optimized for tool calling
            sampling_params = SamplingParams(
                temperature=max(0.3, min(temperature, 0.7)),  # Increase minimum temperature to prevent repetition
                max_tokens=min(max_tokens, 600),  # More tokens for thinking + tool calls
                n=1,
                presence_penalty=0.1,  # Reduce presence penalty to prevent over-constraint
                ignore_eos=False
            )
            
            console.print(Panel(
                f"🌡️ Temperature: {sampling_params.temperature}\n"
                f"📏 Max tokens: {sampling_params.max_tokens}\n"
                f"🔄 Presence penalty: {sampling_params.presence_penalty}",
                title="⚙️ SAMPLING PARAMETERS",
                style="dim",
                border_style="dim"
            ))
            
            # Stream the response using chat format
            accumulated_response = ""
            chunk_count = 0
            
            console.print(Panel(
                "Starting model response generation...",
                title="🚀 MODEL STREAMING STARTED",
                style="bold green",
                border_style="green"
            ))
            
            # Use streaming - force it to use actual streaming
            console.print(Panel(
                "🔄 Attempting to use streaming interface...",
                title="🚀 STREAMING SETUP",
                style="bold cyan"
            ))
            
            # Try chat_stream first
            if hasattr(self.llm, 'chat_stream'):
                console.print("✅ Using chat_stream interface")
                
                # Try to capture the final prompt that will be generated
                if hasattr(self.llm, '_messages_to_prompt'):
                    try:
                        final_prompt = self.llm._messages_to_prompt(chat_messages)
                        self._log_final_processed_prompt(final_prompt)
                    except Exception as e:
                        console.print(f"⚠️ Could not capture final prompt: {e}")
                
                import inspect
                sig = inspect.signature(self.llm.chat_stream)
                if 'sampling_params' in sig.parameters:
                    stream_iterator = self.llm.chat_stream(chat_messages, sampling_params=sampling_params)
                else:
                    stream_iterator = self.llm.chat_stream(chat_messages)
            else:
                # Fallback to prompt-based streaming
                console.print("⚠️ Falling back to prompt-based streaming")
                prompt = self.llm._messages_to_prompt(chat_messages)
                self._log_final_processed_prompt(prompt)
                stream_iterator = self.llm.stream(prompt, sampling_params=sampling_params)
            
            # Check if the iterator is actually async
            if hasattr(stream_iterator, '__aiter__'):
                console.print("✅ Got async iterator for streaming")
            else:
                console.print("⚠️ Got sync iterator for streaming")
            
            # Handle streaming
            if hasattr(stream_iterator, '__aiter__'):
                console.print("🔄 Starting async streaming loop...")
                chunk_number = 0
                # Async generator
                async for chunk in stream_iterator:
                    chunk_number += 1
                    if chunk_number <= 5:  # Log first few chunks
                        console.print(f"📦 Chunk #{chunk_number}: '{chunk}' ({len(chunk)} chars)")
                    elif chunk_number == 6:
                        console.print("📦 ... (continuing to stream, will only log important events)")
                    
                    if websocket.close_code is not None:
                        console.print("⚠️ WebSocket closed during action streaming")
                        break
                    
                    accumulated_response += chunk
                    chunk_count += 1
                    
                    # Check if we hit a tool call that needs user approval
                    if ("<tool_call>" in accumulated_response and 
                        "</tool_call>" in accumulated_response and
                        chunk_count > 5):  # Make sure we have substantial content
                        
                        # Extract and display the tool call
                        tool_call_match = accumulated_response.split('<tool_call>')[1].split('</tool_call>')[0]
                        thinking_match = ""
                        if "<think>" in accumulated_response and "</think>" in accumulated_response:
                            thinking_match = accumulated_response.split('<think>')[1].split('</think>')[0]
                        
                        console.print(Panel(
                            f"🧠 **AI Thinking:**\n{thinking_match}\n\n🛠️ **Tool Call:**\n{tool_call_match}" if thinking_match 
                            else f"🛠️ **Tool Call:**\n{tool_call_match}",
                            title="⏸️ TOOL CALL DETECTED - ENDING MESSAGE",
                            style="bold yellow",
                            border_style="yellow"
                        ))
                        
                        # Stream current content and END the assistant message
                        stream_chunk = {
                            "choices": [{
                                "delta": {
                                    "content": chunk
                                },
                                "finish_reason": "stop"  # End the message here
                            }]
                        }
                        
                        try:
                            await websocket.send(json.dumps(stream_chunk))
                        except websockets.exceptions.ConnectionClosed:
                            console.print(Panel(
                                "WebSocket connection closed while sending tool call chunk",
                                title="⚠️ CONNECTION CLOSED",
                                style="bold red"
                            ))
                            break
                        
                        # Save the assistant message with tool call to task
                        if task_id and accumulated_response:
                            self.task_manager.add_message(task_id, 'assistant', accumulated_response)
                            console.print(Panel(
                                f"💾 Saved assistant message with tool call to task: {task_id}",
                                title="📚 TASK UPDATED",
                                style="bold cyan",
                                border_style="cyan"
                            ))
                        
                        # This will trigger the frontend to show the tool call modal
                        # The streaming will resume when user approves via a separate endpoint
                        return
                    
                    # Stream chunk to client
                    stream_chunk = {
                        "choices": [{
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }]
                    }
                    
                    try:
                        await websocket.send(json.dumps(stream_chunk))
                        await asyncio.sleep(0.02)
                    except websockets.exceptions.ConnectionClosed:
                        console.print("⚠️ WebSocket connection closed while sending action chunk")
                        break
                
                console.print(f"✅ Async streaming completed after {chunk_count} chunks")
            else:
                console.print("🔄 Starting sync streaming loop...")
                chunk_number = 0
                # Regular generator
                for chunk in stream_iterator:
                    chunk_number += 1
                    if chunk_number <= 5:  # Log first few chunks
                        console.print(f"📦 Sync Chunk #{chunk_number}: '{chunk}' ({len(chunk)} chars)")
                    elif chunk_number == 6:
                        console.print("📦 ... (continuing sync stream, will only log important events)")
                    
                    if websocket.close_code is not None:
                        console.print("⚠️ WebSocket closed during action streaming")
                        break
                    
                    accumulated_response += chunk
                    chunk_count += 1
                    
                    # Check for repetitive content and stop if detected
                    if len(accumulated_response) > 100:
                        # Multiple checks for repetitive patterns (similar to garbage filtering)
                        if (accumulated_response.count('"') > 50 or  # Too many quotes
                            accumulated_response.count('</tool_call>') > 2 or  # Repeated closing tags
                            accumulated_response.count('.com') > 15 or  # Repetitive domain patterns
                            accumulated_response.count('://') > 10):  # Too many URLs
                            console.print("🚫 Repetitive garbage content detected, stopping generation")
                            stream_chunk = {
                                "choices": [{
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            try:
                                await websocket.send(json.dumps(stream_chunk))
                            except:
                                pass
                            break
                        
                        # Check for repeated short patterns
                        last_100_chars = accumulated_response[-100:]
                        if len(set(last_100_chars)) < 8:  # Too few unique characters  
                            console.print("⚠️ Repetitive character patterns detected, stopping generation")
                            stream_chunk = {
                                "choices": [{
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            try:
                                await websocket.send(json.dumps(stream_chunk))
                            except:
                                pass
                            break
                    
                    # Check if we hit a tool call that needs user approval
                    if ("<tool_call>" in accumulated_response and 
                        "</tool_call>" in accumulated_response and
                        chunk_count > 5):  # Make sure we have substantial content
                        
                        console.print("⏸️ Tool call detected in sync stream, ending message")
                        
                        # Stream current content and END the assistant message
                        stream_chunk = {
                            "choices": [{
                                "delta": {
                                    "content": chunk
                                },
                                "finish_reason": "stop"  # End the message here
                            }]
                        }
                        
                        try:
                            await websocket.send(json.dumps(stream_chunk))
                        except websockets.exceptions.ConnectionClosed:
                            console.print("⚠️ WebSocket connection closed while sending tool call chunk")
                            break
                        
                        # Save the assistant message with tool call to task
                        if task_id and accumulated_response:
                            self.task_manager.add_message(task_id, 'assistant', accumulated_response)
                            console.print(Panel(
                                f"💾 Saved assistant message with tool call to task: {task_id}",
                                title="📚 TASK UPDATED",
                                style="bold cyan",
                                border_style="cyan"
                            ))
                        
                        # This will trigger the frontend to show the tool call modal
                        return
                    
                    # Stream chunk to client
                    stream_chunk = {
                        "choices": [{
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }]
                    }
                    
                    try:
                        await websocket.send(json.dumps(stream_chunk))
                        await asyncio.sleep(0.02)
                    except websockets.exceptions.ConnectionClosed:
                        console.print("⚠️ WebSocket connection closed while sending action chunk")
                        break
                
                console.print(f"✅ Sync streaming completed after {chunk_count} chunks")
            
            # If we reach here, no tool call was detected, send final chunk
            if websocket.close_code is None:
                final_chunk = {
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                
                await websocket.send(json.dumps(final_chunk))
                
                # Log the final response
                self._log_model_response(accumulated_response, "FINAL")
                
                console.print(Panel(
                    f"🎯 Tool-enabled response completed\n"
                    f"📏 Length: {len(accumulated_response)} characters\n"
                    f"📊 Total chunks: {chunk_count}",
                    title="✅ ACTION STREAMING COMPLETED",
                    style="bold green",
                    border_style="green"
                ))
                
                # Save to task if provided
                if task_id and accumulated_response:
                    self.task_manager.add_message(task_id, 'assistant', accumulated_response)
                    
                    console.print(Panel(
                        f"💾 Saved tool response to task: {task_id}",
                        title="📚 TASK UPDATED",
                        style="bold cyan",
                        border_style="cyan"
                    ))
            
        except Exception as e:
            console.print(Panel(
                f"❌ Error in action streaming: {str(e)}",
                title="🚨 ACTION STREAMING ERROR",
                style="bold red",
                border_style="red"
            ))
            import traceback
            traceback.print_exc()
            
            if websocket.close_code is None:
                try:
                    error_chunk = {
                        "choices": [{
                            "delta": {
                                "content": f"\n\n[Action Error: {str(e)}]"
                            },
                            "finish_reason": "stop"
                        }]
                    }
                    await websocket.send(json.dumps(error_chunk))
                except:
                    pass

    async def continue_task_conversation(self, websocket, task_id: str):
        """Continue the conversation for a task until 'end' action is reached"""
        try:
            logger.info(f"🐛 continue_task_conversation called with task_id={task_id}")
            
            # Get current task messages
            task_data = self.get_task(task_id)
            if task_data.get('error'):
                console.print(Panel(
                    f"❌ Task not found: {task_id}",
                    title="🚨 TASK NOT FOUND",
                    style="bold red",
                    border_style="red"
                ))
                return
            
            messages = task_data.get('messages', [])
            task_model = task_data.get('task', {}).get('model', 'qwen2.5-vl')
            mentioned_servers = task_data.get('task', {}).get('mcp_servers', [])
            
            console.print(Panel(
                f"🔄 **Continuing task conversation**\n"
                f"🎯 **Task ID:** {task_id}\n"
                f"🤖 **Model:** {task_model}\n"
                f"📊 **Message count:** {len(messages)}\n"
                f"🛠️ **MCP Servers:** {mentioned_servers}",
                title="🚀 TASK CONTINUATION",
                style="bold cyan",
                border_style="cyan"
            ))
            
            # Build message history for the model
            message_history = []
            logger.info(f"🐛 Building message history from {len(messages)} messages")
            
            for i, msg in enumerate(messages):
                logger.info(f"🐛 Processing message {i}: {msg['role']} - {msg['content'][:100]}...")
                
                # Skip obviously broken assistant messages (repetitive garbage)
                if msg['role'] == 'assistant':
                    content = msg['content']
                    # Check for repetitive patterns that indicate garbage output
                    if (content.count('"') > 20 or  # Too many quotes
                        content.count('</tool_call>') > 2 or  # Repeated closing tags
                        content.count('.com') > 10 or  # Repetitive domain patterns
                        len(content) > 1000 and len(set(content.split())) < 10):  # Very repetitive words
                        logger.warning(f"⚠️ Skipping garbage assistant message: {content[:100]}...")
                        continue
                
                # Handle different message types
                if msg['role'] == 'tool':
                    # Skip 'tool' role messages - they're now handled as structured user messages
                    # or convert legacy tool responses for backward compatibility
                    if len(message_history) > 0 and message_history[-1]['role'] == 'user' and '<tool_response>' in message_history[-1]['content']:
                        # Tool response already included as structured user message, skip the 'tool' role duplicate
                        logger.debug(f"⏭️ Skipping tool message - already included as structured user message")
                        continue
                    else:
                        # Legacy tool response - convert for backward compatibility
                        try:
                            tool_result = json.loads(msg['content'])
                            if tool_result.get('success', True):
                                content_str = str(tool_result.get('content', tool_result))
                                if len(content_str) > 1000:
                                    content_str = content_str[:1000] + "... [truncated]"
                                context_msg = f"<tool_response>\n{content_str}\n</tool_response>"
                            else:
                                context_msg = f"<tool_response>\nTool execution failed: {tool_result.get('error', 'Unknown error')}\n</tool_response>"
                        except:
                            tool_content = msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content']
                            context_msg = f"<tool_response>\n{tool_content}\n</tool_response>"
                        
                        message_history.append({
                            'role': 'user',
                            'content': context_msg
                        })
                        logger.debug(f"🔄 Converted legacy tool response to structured user message: {len(context_msg)} chars")
                else:
                    # Include regular messages as-is
                    message_history.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
                    logger.debug(f"➕ Added {msg['role']} message: {len(msg['content'])} chars")
            
            logger.info(f"🐛 Final message history has {len(message_history)} messages:")
            for i, msg in enumerate(message_history):
                logger.info(f"🐛   {i}: {msg['role']} - {msg['content'][:200]}...")
            
            # Check if the last message indicates we should end
            # Look for specific end patterns, not just any occurrence of "end"
            end_patterns = [
                r'<tool_call>\s*\{\s*"name":\s*"end"',  # end tool call
                r'action.*end',  # end action
                r'task.*complet',  # task complete/completed
                r'finished.*task',  # finished task
                r'end.*task',  # end task
                r'stop.*here',  # stop here
                r'done.*with.*task'  # done with task
            ]
            
            import re
            should_end = False
            if messages:
                for msg in messages[-3:]:
                    if msg['role'] == 'assistant':
                        content_lower = msg['content'].lower()
                        for pattern in end_patterns:
                            if re.search(pattern, content_lower):
                                should_end = True
                                break
                        if should_end:
                            break
            
            if should_end:
                console.print(Panel(
                    "🎯 Task appears to be completed (detected end pattern)",
                    title="✅ TASK COMPLETED",
                    style="bold green",
                    border_style="green"
                ))
                return
            
            # Continue working on the task - agent should make more tool calls until 'end' action
            

            
            # Ensure we have enough context to continue
            if len(message_history) < 2:
                logger.warning("⚠️ Insufficient message history for continuation, stopping")
                return
            
            # Log the message history being sent for debugging
            logger.info(f"📋 Sending {len(message_history)} messages to model for continuation:")
            for i, msg in enumerate(message_history[-3:]):  # Show last 3 messages
                content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                logger.info(f"  {i}: {msg['role']} - {content_preview}")
            
            # Continue the conversation using stream_response
            logger.info(f"🐛 Calling stream_response for task continuation")
            logger.info(f"🐛 Message history length: {len(message_history)}")
            logger.info(f"🐛 Task model: {task_model}")
            await self.stream_response(
                websocket=websocket,
                messages=message_history,
                model=task_model,
                temperature=0.5,  # Slightly higher temperature to encourage generation
                max_tokens=500,   # Increased token limit for better responses
                task_id=task_id
            )
            
        except Exception as e:
            console.print(Panel(
                f"❌ Error continuing task conversation: {str(e)}",
                title="🚨 TASK CONTINUATION ERROR",
                style="bold red",
                border_style="red"
            ))
            import traceback
            traceback.print_exc()

async def handle_client(websocket):
    """Handle WebSocket client connections"""
    client_addr = websocket.remote_address
    
    console.print(Panel(
        f"🌐 Client connected: {client_addr[0]}:{client_addr[1]}",
        title="🔗 NEW CONNECTION",
        style="bold blue",
        border_style="blue"
    ))
    
    # Use global LLM server instance (loaded at startup)
    llm_server = global_llm_server
    
    # Fallback: create new instance if global server failed to initialize
    if llm_server is None:
        console.print(Panel(
            "⚠️ Global model not available, creating new instance...",
            title="🔄 FALLBACK INITIALIZATION",
            style="bold yellow",
            border_style="yellow"
        ))
    llm_server = DarkRLLLMServer()
    
    try:
        async for message in websocket:
            try:
                # Parse request
                request = json.loads(message)
                logger.info(f"🔥 RECEIVED REQUEST from {client_addr}: type={request.get('type', 'unknown')}")
                logger.debug(f"🔥 FULL REQUEST: {json.dumps(request, indent=2)[:500]}...")
                
                # Handle MCP server list requests
                if request.get('type') == 'list_mcp_servers':
                    query = request.get('query', '')
                    servers = llm_server.get_mcp_servers(query)
                    
                    response = {
                        "type": "mcp_servers_response",
                        "servers": servers
                    }
                    await websocket.send(json.dumps(response))
                    continue

                # Handle MCP server actions requests
                if request.get('type') == 'get_mcp_server_actions':
                    server_ids = request.get('server_ids', [])
                    server_actions = await llm_server.get_mcp_server_actions(server_ids)
                    
                    # Send response for each server
                    for server_id, actions in server_actions.items():
                        response = {
                            "type": "mcp_server_actions_response",
                            "server_id": server_id,
                            "actions": actions
                        }
                        await websocket.send(json.dumps(response))
                    continue
                
                # Handle MCP action execution requests
                if request.get('type') == 'execute_mcp_action':
                    server_id = request.get('server_id', '')
                    action_name = request.get('action_name', '')
                    parameters = request.get('parameters', {})
                    
                    if not server_id or not action_name:
                        error_response = {
                            "type": "error",
                            "error": {"message": "server_id and action_name are required", "type": "invalid_request"}
                        }
                        await websocket.send(json.dumps(error_response))
                        continue
                    
                    logger.info(f"Executing MCP action {action_name} on server {server_id} with parameters: {parameters}")
                    
                    result = await llm_server.execute_mcp_action(server_id, action_name, parameters)
                    
                    response = {
                        "type": "mcp_action_result",
                        "server_id": server_id,
                        "action_name": action_name,
                        "parameters": parameters,
                        **result
                    }
                    await websocket.send(json.dumps(response))
                    continue
                
                # Handle API key management requests
                if request.get('type') == 'get_api_keys':
                    try:
                        keys = load_local_api_keys()
                        response = {
                            "type": "api_keys_response",
                            "success": True,
                            "keys": keys
                        }
                        await websocket.send(json.dumps(response))
                    except Exception as e:
                        response = {
                            "type": "api_keys_response", 
                            "success": False,
                            "error": str(e)
                        }
                        await websocket.send(json.dumps(response))
                    continue
                
                if request.get('type') == 'save_api_keys':
                    try:
                        keys = request.get('keys', {})
                        # Save to local file
                        agentsea_dir = Path.home() / '.agentsea'
                        agentsea_dir.mkdir(exist_ok=True)
                        api_keys_file = agentsea_dir / 'api_keys.json'
                        
                        with open(api_keys_file, 'w') as f:
                            json.dump(keys, f, indent=2)
                        
                        # Clear MCP connection cache so connections are recreated with new API keys
                        await llm_server.cleanup_mcp_connections()
                        
                        response = {
                            "type": "api_keys_response",
                            "success": True
                        }
                        await websocket.send(json.dumps(response))
                    except Exception as e:
                        response = {
                            "type": "api_keys_response",
                            "success": False,
                            "error": str(e)
                        }
                        await websocket.send(json.dumps(response))
                    continue
                
                # Handle MCP actions cache refresh
                if request.get('type') == 'refresh_mcp_actions':
                    server_id = request.get('server_id')
                    if not server_id:
                        error_response = {
                            "type": "error",
                            "error": {"message": "server_id is required", "type": "invalid_request"}
                        }
                        await websocket.send(json.dumps(error_response))
                        continue
                    
                    logger.info(f"Refreshing MCP actions for server {server_id}")
                    result = await llm_server.refresh_mcp_actions(server_id)
                    
                    response = {
                        "type": "mcp_actions_refreshed",
                        "server_id": server_id,
                        "actions": result.get(server_id, [])
                    }
                    await websocket.send(json.dumps(response))
                    continue
                
                # Handle MCP actions cache status
                if request.get('type') == 'get_mcp_cache_status':
                    cache_stats = llm_server.get_mcp_cache_stats()
                    
                    response = {
                        "type": "mcp_cache_status",
                        **cache_stats
                    }
                    await websocket.send(json.dumps(response))
                    continue
                
                # Handle tool execution and stream continuation
                if request.get('type') == 'execute_tool_and_continue':
                    server_id = request.get('server_id', '')
                    tool_name = request.get('tool_name', '')
                    parameters = request.get('parameters', {})
                    current_response = request.get('current_response', '')
                    mentioned_servers = request.get('mentioned_servers', [])
                    model = request.get('model', 'qwen2.5-vl')
                    task_id = request.get('task_id')
                    
                    if not server_id or not tool_name:
                        error_response = {
                            "type": "error",
                            "error": {"message": "server_id and tool_name are required", "type": "invalid_request"}
                        }
                        await websocket.send(json.dumps(error_response))
                        continue
                    
                    console.print(Panel(
                        f"🚀 Executing tool: **{tool_name}**\n"
                        f"📡 Server: **{server_id}**\n"
                        f"⚙️ Parameters: `{json.dumps(parameters, indent=2)}`",
                        title="🔧 TOOL EXECUTION STARTED",
                        style="bold blue",
                        border_style="blue"
                    ))
                    
                    try:
                        # Execute the tool
                        action_result = await llm_server.execute_mcp_action(server_id, tool_name, parameters)
                        
                        console.print(Panel(
                            f"✅ Tool executed successfully\n\n"
                            f"**Result:**\n```json\n{json.dumps(action_result, indent=2)}\n```",
                            title="🎉 TOOL EXECUTION COMPLETED",
                            style="bold green",
                            border_style="green"
                        ))
                        
                        # Save tool response in structured format as user message
                        tool_response_content = json.dumps(action_result, indent=2)
                        structured_response = f"<tool_response>\n{tool_response_content}\n</tool_response>"
                        llm_server.task_manager.add_message(task_id, 'user', structured_response)
                        
                        console.print(Panel(
                            f"💾 **Tool response saved as user message**\n\n"
                            f"**Result:**\n```json\n{tool_response_content}\n```\n\n"
                            f"**Saved as:** `<tool_response>...</tool_response>`",
                            title="🔧 TOOL RESPONSE SAVED",
                            style="bold green",
                            border_style="green"
                        ))
                        
                        # Stream just a completion message to close the tool execution
                        tool_response_chunk = {
                            "choices": [{
                                "delta": {
                                    "content": "\n\n✅ Tool executed successfully."
                                },
                                "finish_reason": "stop"  # End the message
                            }]
                        }
                        
                        await websocket.send(json.dumps(tool_response_chunk))
                        
                        # Now generate the final assistant response after the tool execution
                        # Build a fresh context for the final response (without the complex tool context)
                        # We'll generate a simple concluding response based on the tool result
                        
                        # Create a simple prompt for the final response
                        conclusion_prompt = f"Based on the tool execution result, provide a brief summary or conclusion to the user."
                        
                        chat_messages = [
                            {"role": "user", "content": conclusion_prompt},
                        ]
                        
                        # Create sampling parameters for continuation
                        sampling_params = SamplingParams(
                            temperature=0.3,
                            max_tokens=400,  # Remaining tokens for conclusion
                            n=1,
                            presence_penalty=0.1,
                            ignore_eos=False
                        )
                        
                        console.print(Panel(
                            "🔄 Continuing model generation after tool execution...",
                            title="↩️ RESUMING STREAM",
                            style="bold cyan",
                            border_style="cyan"
                        ))
                        
                        # Continue streaming from where we left off
                        if hasattr(llm_server.llm, 'chat_stream'):
                            import inspect
                            sig = inspect.signature(llm_server.llm.chat_stream)
                            if 'sampling_params' in sig.parameters:
                                stream_iterator = llm_server.llm.chat_stream(chat_messages, sampling_params=sampling_params)
                            else:
                                stream_iterator = llm_server.llm.chat_stream(chat_messages)
                        else:
                            prompt = llm_server.llm._messages_to_prompt(chat_messages)
                            stream_iterator = llm_server.llm.stream(prompt, sampling_params=sampling_params)
                        
                        # Continue streaming the response
                        continuation_response = ""
                        if hasattr(stream_iterator, '__aiter__'):
                            async for chunk in stream_iterator:
                                if websocket.close_code is not None:
                                    break
                                
                                continuation_response += chunk
                                
                                stream_chunk = {
                                    "choices": [{
                                        "delta": {
                                            "content": chunk
                                        },
                                        "finish_reason": None
                                    }]
                                }
                                
                                try:
                                    await websocket.send(json.dumps(stream_chunk))
                                    await asyncio.sleep(0.02)
                                except websockets.exceptions.ConnectionClosed:
                                    break
                        else:
                            for chunk in stream_iterator:
                                if websocket.close_code is not None:
                                    break
                                
                                continuation_response += chunk
                                
                                stream_chunk = {
                                    "choices": [{
                                        "delta": {
                                            "content": chunk
                                        },
                                        "finish_reason": None
                                    }]
                                }
                                
                                try:
                                    await websocket.send(json.dumps(stream_chunk))
                                    await asyncio.sleep(0.02)
                                except websockets.exceptions.ConnectionClosed:
                                    break
                        
                        # Send final chunk
                        if websocket.close_code is None:
                            final_chunk = {
                                "choices": [{
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            
                            await websocket.send(json.dumps(final_chunk))
                            
                            # Save final conclusion response to task
                            if task_id and continuation_response:
                                llm_server.task_manager.add_message(task_id, 'assistant', continuation_response)
                                
                                console.print(Panel(
                                    f"💾 Saved final response to task: {task_id}\n"
                                    f"📏 Response length: {len(continuation_response)} characters",
                                    title="✅ FINAL RESPONSE SAVED",
                                    style="bold green",
                                    border_style="green"
                                ))
                    
                    except Exception as e:
                        console.print(Panel(
                            f"❌ Error executing tool and continuing: {str(e)}",
                            title="🚨 TOOL EXECUTION ERROR",
                            style="bold red",
                            border_style="red"
                        ))
                        
                        error_chunk = {
                            "choices": [{
                                "delta": {
                                    "content": f"\n\n❌ **Error executing tool:** {str(e)}"
                                },
                                "finish_reason": "stop"
                            }]
                        }
                        await websocket.send(json.dumps(error_chunk))
                    
                    continue
                
                # Handle task creation requests
                if request.get('type') == 'create_task':
                    logger.info(f"Received create_task request: {request}")
                    initial_prompt = request.get('prompt', '')
                    model = request.get('model', 'qwen2.5-vl')
                    
                    if not initial_prompt:
                        logger.error("No prompt provided in create_task request")
                        error_response = {
                            "type": "error",
                            "error": {"message": "No prompt provided", "type": "invalid_request"}
                        }
                        await websocket.send(json.dumps(error_response))
                        continue
                    
                    logger.info(f"Creating task with prompt: {initial_prompt[:100]}...")
                    task_id = llm_server.create_task(initial_prompt, model)
                    logger.info(f"Task created with ID: {task_id}")
                    
                    response = {
                        "type": "task_created",
                        "task_id": task_id
                    }
                    await websocket.send(json.dumps(response))
                    logger.info(f"Sent task_created response to client")
                    continue
                
                # Handle task retrieval requests
                if request.get('type') == 'get_task':
                    logger.info(f"Received get_task request: {request}")
                    task_id = request.get('task_id', '')
                    
                    if not task_id:
                        logger.error("No task_id provided in get_task request")
                        error_response = {
                            "type": "error",
                            "error": {"message": "No task_id provided", "type": "invalid_request"}
                        }
                        await websocket.send(json.dumps(error_response))
                        continue
                    
                    logger.info(f"Retrieving task with ID: {task_id}")
                    task_data = llm_server.get_task(task_id)
                    logger.info(f"Task data retrieved: {task_data}")
                    
                    response = {
                        "type": "task_data",
                        **task_data
                    }
                    await websocket.send(json.dumps(response))
                    logger.info(f"Sent task_data response to client")
                    continue
                
                # Handle recent tasks requests
                if request.get('type') == 'get_recent_tasks':
                    tasks = llm_server.get_recent_tasks()
                    response = {
                        "type": "recent_tasks",
                        "tasks": tasks
                    }
                    await websocket.send(json.dumps(response))
                    continue

                # Handle enhanced correction requests (NEW SIMPLIFIED WORKFLOW)
                if request.get('type') == 'correction_with_execution':
                    console.print(Panel(
                        f"🔧 Received correction request",
                        title="📝 CORRECTION REQUEST",
                        style="bold yellow",
                        border_style="yellow"
                    ))
                    
                    try:
                        task_id = request.get('task_id')
                        message_index = request.get('message_index')  # Index of bad response to replace
                        corrected_tool_call = request.get('corrected_tool_call')  # {"name": "tool", "arguments": {...}}
                        thought = request.get('thought', '')  # Reasoning for the correction
                        
                        if not task_id or corrected_tool_call is None:
                            error_response = {
                                "type": "error",
                                "error": {"message": "task_id and corrected_tool_call are required", "type": "invalid_request"}
                            }
                            await websocket.send(json.dumps(error_response))
                            continue
                        
                        console.print(Panel(
                            f"🎯 **Task ID:** {task_id}\n"
                            f"🔧 **Corrected Tool:** {corrected_tool_call.get('name', 'unknown')}\n"
                            f"⚙️ **Parameters:** {json.dumps(corrected_tool_call.get('arguments', {}), indent=2)}\n"
                            f"🧠 **Thought:** {thought}",
                            title="📋 CORRECTION DETAILS",
                            style="bold cyan",
                            border_style="cyan"
                        ))
                        
                        # Build the corrected response with think tags
                        corrected_response = ""
                        if thought:
                            corrected_response += f"<think>\n{thought}\n</think>\n\n"
                        
                        # Extract the actual tool name without server prefix for the corrected response
                        tool_name_full = corrected_tool_call.get('name')
                        if '.' in tool_name_full:
                            _, actual_tool_name = tool_name_full.split('.', 1)
                        else:
                            actual_tool_name = tool_name_full
                        
                        # Use the unprefixed tool name in the saved response
                        corrected_tool_call_clean = {
                            "name": actual_tool_name,
                            "arguments": corrected_tool_call.get('arguments', {})
                        }
                        
                        corrected_response += f"<tool_call>\n{json.dumps(corrected_tool_call_clean, indent=2)}\n</tool_call>"
                        
                        console.print(Panel(
                            corrected_response,
                            title="✅ CORRECTED RESPONSE",
                            style="bold green", 
                            border_style="green"
                        ))
                        
                        # 1. Replace the bad response in the task
                        task_data = llm_server.get_task(task_id)
                        success = False
                        if not task_data.get('error'):
                            success = llm_server.task_manager.replace_message(task_id, message_index, corrected_response)
                            
                            if success:
                                console.print(Panel(
                                    f"💾 **Replaced bad response** in task {task_id}\n"
                                    f"📍 **Message index:** {message_index}\n"
                                    f"📝 **New length:** {len(corrected_response)} chars",
                                    title="🔄 RESPONSE REPLACED",
                                    style="bold purple",
                                    border_style="purple"
                                ))
                        
                        # 2. Apply learning from the correction
                        if task_data and not task_data.get('error'):
                            messages = task_data.get('messages', [])
                            # Find the user prompt that led to the bad response
                            user_prompt = None
                            if message_index > 0:
                                for i in range(message_index - 1, -1, -1):
                                    if messages[i]['role'] == 'user':
                                        user_prompt = messages[i]['content']
                                        break
                            
                            if user_prompt:
                                # Ensure we're using the same model as the task
                                task_model = task_data.get('task', {}).get('model', 'qwen2.5-vl')
                                if llm_server.current_model != task_model:
                                    logger.info(f"Switching model for correction learning: {llm_server.current_model} -> {task_model}")
                                    await llm_server.initialize_model(task_model)
                                
                                # Create learning example with the corrected response
                                learning_example = [
                                    {"role": "user", "content": user_prompt},
                                    {"role": "assistant", "content": corrected_response}
                                ]
                                
                                # Pretty print the correction learning example
                                console.print(Panel(
                                    f"📚 **Learning from CORRECTED RESPONSE**\n\n"
                                    f"**JSON Messages:**\n```json\n{json.dumps(learning_example, indent=2)}\n```\n\n"
                                    f"**Processed as Plain Text:**\n"
                                    f"👤 **User:** {user_prompt}\n\n"
                                    f"🤖 **Corrected Assistant:** {corrected_response}",
                                    title="🔧 CORRECTIVE LEARNING EXAMPLE",
                                    style="bold cyan",
                                    border_style="cyan"
                                ))
                                
                                # Apply corrective learning
                                await llm_server.llm.learn(
                                    learning_example,
                                    adapter="correction_learning",
                                    steps=15,  # More intensive for corrections
                                    lr=2e-4   # Higher learning rate for corrections
                                )
                                
                                console.print(Panel(
                                    "🧠 Applied corrective learning to model",
                                    title="🎓 LEARNING COMPLETED",
                                    style="bold magenta",
                                    border_style="magenta"
                                ))
                        
                        # 3. Execute the corrected tool
                        tool_name_full = corrected_tool_call.get('name')
                        parameters = corrected_tool_call.get('arguments', {})
                        
                        # Extract server_id and tool_name from the full tool name (e.g., "playwright.browser_navigate")
                        if '.' in tool_name_full:
                            server_id, tool_name = tool_name_full.split('.', 1)
                            logger.info(f"🔧 Extracted server_id: {server_id}, tool_name: {tool_name}")
                        else:
                            # Fallback: try to find the server by searching
                            tool_name = tool_name_full
                            server_id = None
                            server_actions = await llm_server.get_mcp_server_actions(['playwright', 'filesystem', 'sequential-thinking'])
                            for sid, actions in server_actions.items():
                                for action in actions:
                                    if action.get('name') == tool_name:
                                        server_id = sid
                                        break
                                if server_id:
                                    break
                            logger.info(f"🔍 Fallback search found server_id: {server_id}, tool_name: {tool_name}")
                        
                        if server_id:
                            logger.info(f"🚀 Executing corrected tool: {tool_name} on server {server_id}")
                            execution_result = await llm_server.execute_mcp_action(server_id, tool_name, parameters)
                            logger.info(f"✅ Corrected tool execution completed")
                            
                            # Test JSON serialization before saving
                            try:
                                tool_response_content = json.dumps(execution_result, indent=2)
                                logger.debug("✅ Corrected tool result is JSON serializable")
                            except Exception as json_error:
                                logger.error(f"❌ Corrected tool result is NOT JSON serializable: {json_error}")
                                # Fallback to string representation
                                tool_response_content = str(execution_result)
                                logger.warning("🔄 Using string fallback for corrected tool result")
                            
                            # 4. Add structured tool response as user message
                            structured_response = f"<tool_response>\n{tool_response_content}\n</tool_response>"
                            llm_server.task_manager.add_message(task_id, 'user', structured_response)
                            logger.debug(f"💾 Added corrected tool response as user message: {len(structured_response)} characters")
                            
                            console.print(Panel(
                                f"✅ **Corrected tool executed**\n\n"
                                f"**Result:**\n```json\n{tool_response_content}\n```\n\n"
                                f"**Saved as user message:** `<tool_response>...</tool_response>`",
                                title="🎉 EXECUTION COMPLETED",
                                style="bold green",
                                border_style="green"
                            ))
                            
                            # 5. Continue the task conversation
                            await llm_server.continue_task_conversation(websocket, task_id)
                        else:
                            logger.error(f"❌ Could not find server for corrected tool: {tool_name}")
                            console.print(Panel(
                                f"❌ Could not find server for tool: {tool_name}",
                                title="🚨 EXECUTION ERROR",
                                style="bold red",
                                border_style="red"
                            ))
                        
                        # Send response back to frontend
                        response = {
                            'type': 'correction_processed',
                            'success': True,
                            'corrected_response': corrected_response,
                            'message_replaced': success
                        }
                        await websocket.send(json.dumps(response))
                        
                    except Exception as e:
                        console.print(Panel(
                            f"❌ Error processing correction: {str(e)}",
                            title="🚨 CORRECTION ERROR",
                            style="bold red",
                            border_style="red"
                        ))
                        
                        error_response = {
                            'type': 'error',
                            'error': f'Failed to process correction: {str(e)}'
                        }
                        await websocket.send(json.dumps(error_response))
                    
                    continue

                # Handle model selection from dual response - SIMPLIFIED
                if request.get('type') == 'model_selection':
                    try:
                        task_id = request.get('task_id', '')
                        selected_model = request.get('selected_model', '')  # 'local' or 'gpt'
                        
                        # NEW: Get the selected content directly from the request (streaming approach)
                        local_response = request.get('local_response', '')
                        gpt_response = request.get('gpt_response', '')
                        
                        console.print(Panel(
                            f"🔥 MODEL SELECTION REQUEST RECEIVED!\n"
                            f"📍 **Client:** {client_addr}\n"
                            f"🎯 **Task ID:** {task_id}\n"
                            f"🔹 **Selected Model:** {selected_model}\n"
                            f"📊 **Local Response Length:** {len(local_response)} chars\n"
                            f"📊 **GPT Response Length:** {len(gpt_response)} chars",
                            title="🎪 USER MODEL SELECTION",
                            style="bold blue",
                            border_style="blue"
                        ))
                        
                        # Get selected content from the request (not from database)
                        selected_content = local_response if selected_model == 'local' else gpt_response
                        
                        if not selected_content:
                            error_response = {
                                "type": "model_selection_response",
                                "success": False,
                                "error": f"No {selected_model} response content found"
                            }
                            await websocket.send(json.dumps(error_response))
                            continue
                        
                        # Save the selected response to the task
                        if task_id and selected_model and selected_content:
                            llm_server.task_manager.add_message(task_id, 'assistant', selected_content)
                            logger.info(f"💾 Saved selected {selected_model} response to task {task_id}")
                            
                            # Save user preference for analytics and learning
                            try:
                                # Get the last user message to associate with this preference
                                task_data = llm_server.get_task(task_id)
                                if task_data and not task_data.get('error'):
                                    messages = task_data.get('messages', [])
                                    user_prompt = ""
                                    for msg in reversed(messages):
                                        if msg['role'] == 'user':
                                            user_prompt = msg['content']
                                            break
                                    
                                    if user_prompt:
                                        preference_id = llm_server.task_manager.save_response_preference(
                                            task_id=task_id,
                                            user_prompt=user_prompt,
                                            local_response=local_response,
                                            gpt_response=gpt_response, 
                                            preferred_model=selected_model,
                                            local_model_name=request.get('local_model_name', 'qwen3'),
                                            gpt_model_name=request.get('gpt_model_name', 'gpt-4.1')
                                        )
                                        logger.info(f"📊 Saved response preference {preference_id}: user preferred {selected_model}")
                            except Exception as e:
                                logger.error(f"❌ Error saving response preference: {e}")
                        
                        # Set task state to processing (not idle yet)
                        llm_server.task_manager.set_task_state(task_id, 'processing')
                        
                        # ALWAYS add the selected response to the task messages first
                        console.print(Panel(
                            f"📝 Adding selected response to task messages\\n"
                            f"🤖 Model: {selected_model}\\n"
                            f"📊 Content length: {len(selected_content)} chars",
                            title="💾 SAVING SELECTED RESPONSE",
                            style="bold green",
                            border_style="green"
                        ))
                        llm_server.task_manager.add_message(
                            task_id, 
                            'assistant', 
                            selected_content
                        )
                        
                        # Send confirmation response
                        selection_response = {
                            "type": "model_selection_response",
                            "success": True,
                            "selected_model": selected_model,
                            "task_id": task_id
                        }
                        
                        # Check if the selected response contains a tool call
                        if "<tool_call>" in selected_content and "</tool_call>" in selected_content:
                            logger.info(f"🐛 Selected response contains tool call - will execute and continue conversation")
                            
                            try:
                                tool_call_match = selected_content.split('<tool_call>')[1].split('</tool_call>')[0]
                                tool_call_data = json.loads(tool_call_match)
                                
                                # Find which server this tool belongs to
                                server_id = None
                                server_actions = await llm_server.get_mcp_server_actions(['playwright', 'filesystem', 'sequential-thinking'])
                                
                                for sid, actions in server_actions.items():
                                    for action in actions:
                                        if action.get('name') == tool_call_data.get('name'):
                                            server_id = sid
                                            break
                                    if server_id:
                                        break
                                
                                if server_id:
                                    # Execute the tool
                                    execution_result = await llm_server.execute_mcp_action(
                                        server_id, 
                                        tool_call_data.get('name'), 
                                        tool_call_data.get('arguments', {})
                                    )
                                    
                                    # Create structured tool response
                                    tool_response_content = json.dumps(execution_result, indent=2)
                                    structured_response = f"<tool_response>\n{tool_response_content}\n</tool_response>"
                                    
                                    # Add tool response as user message
                                    llm_server.task_manager.add_message(task_id, 'user', structured_response)
                                    
                                    # Set task state to idle AFTER tool execution
                                    console.print(Panel(
                                        f"✅ Tool execution completed, setting state to 'idle'",
                                        title="📊 STATE CHANGE",
                                        style="bold green",
                                        border_style="green"
                                    ))
                                    llm_server.task_manager.set_task_state(task_id, 'idle')
                                    
                                    # Send tool result to client for seamless UI update
                                    await websocket.send(json.dumps({
                                        "type": "tool_result",
                                        "task_id": task_id,
                                        "message": {
                                            "role": "user",
                                            "content": structured_response,
                                            "timestamp": datetime.datetime.now().isoformat()
                                        }
                                    }))
                                    
                                else:
                                    logger.error(f"❌ Could not find server for tool: {tool_call_data.get('name')}")
                                    # Set task state to idle even if tool execution failed
                                    llm_server.task_manager.set_task_state(task_id, 'idle')
                                    
                            except Exception as tool_error:
                                logger.error(f"❌ Error executing tool from selected response: {tool_error}")
                                # Set task state to idle even if tool execution failed
                                llm_server.task_manager.set_task_state(task_id, 'idle')
                        else:
                            # No tool call, set task state to idle
                            console.print(Panel(
                                f"💬 No tool call found in selected response, setting state to 'idle'",
                                title="📊 STATE CHANGE",
                                style="bold blue",
                                border_style="blue"
                            ))
                            llm_server.task_manager.set_task_state(task_id, 'idle')
                            
                        # Send confirmation LAST, after all processing is done.
                        # The frontend will use this to trigger a reload.
                        await websocket.send(json.dumps(selection_response))
                        
                    except Exception as e:
                        logger.error(f"Error processing model selection: {e}")
                        error_response = {
                            "type": "model_selection_response",
                            "success": False,
                            "error": str(e)
                        }
                        await websocket.send(json.dumps(error_response))
                    continue
                
                # Handle learning feedback requests (NEW SIMPLIFIED WORKFLOW)
                if request.get('type') == 'learning_feedback':
                    logger.info(f"Received learning_feedback request: {request}")
                    
                    feedback_type = request.get('feedback_type')
                    message = request.get('message')
                    task_id = request.get('task_id')
                    user_comment = request.get('user_comment')
                    message_index = request.get('message_index')
                    
                    logger.info(f"Learning feedback: {feedback_type} for task {task_id}")
                    
                    try:
                        # Get task and message history for context
                        task_data = llm_server.get_task(task_id)
                        if task_data.get('error'):
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'error': 'Task not found'
                            }))
                            continue
                        
                        # Ensure we're using the same model as the task
                        task_model = task_data.get('task', {}).get('model', 'qwen2.5-vl')
                        if llm_server.current_model != task_model:
                            logger.info(f"Switching model for learning: {llm_server.current_model} -> {task_model}")
                            await llm_server.initialize_model(task_model)
                        else:
                            logger.info(f"Using existing model for learning: {task_model}")
                        
                        messages = task_data.get('messages', [])
                        
                        if feedback_type == 'approve':
                            # 1. Train on the approved example
                            user_prompt = None
                            for i, msg in enumerate(messages):
                                if msg['role'] == 'assistant' and msg['content'] == message['content']:
                                    # Find the previous user message
                                    for j in range(i-1, -1, -1):
                                        if messages[j]['role'] == 'user':
                                            user_prompt = messages[j]['content']
                                            break
                                    break
                            
                            if user_prompt:
                                learning_example = [
                                    {"role": "user", "content": user_prompt},
                                    {"role": "assistant", "content": message['content']}
                                ]
                                
                                console.print(Panel(
                                    f"📚 **Learning from APPROVED response**\n\n"
                                    f"**JSON Messages:**\n```json\n{json.dumps(learning_example, indent=2)}\n```\n\n"
                                    f"**Processed as Plain Text:**\n"
                                    f"👤 **User:** {user_prompt}\n\n"
                                    f"🤖 **Assistant:** {message['content']}",
                                    title="✅ POSITIVE LEARNING EXAMPLE",
                                    style="bold green",
                                    border_style="green"
                                ))
                                
                                await llm_server.llm.learn(
                                    learning_example,
                                    adapter="task_learning",
                                    steps=10,
                                    lr=1e-4
                                )
                                
                            # 2. Check if there's a tool call to execute
                            tool_call_executed = False
                            if "<tool_call>" in message['content'] and "</tool_call>" in message['content']:
                                try:
                                    console.print(Panel(
                                        "🔍 Parsing tool call from approved message...",
                                        title="🔧 TOOL CALL PARSING",
                                        style="bold yellow",
                                        border_style="yellow"
                                    ))
                                    
                                    tool_call_match = message['content'].split('<tool_call>')[1].split('</tool_call>')[0]
                                    logger.debug(f"🔍 Extracted tool call JSON: {tool_call_match}")
                                    
                                    tool_call_data = json.loads(tool_call_match)
                                    logger.info(f"✅ Parsed tool call: {tool_call_data}")
                                    
                                    console.print(Panel(
                                        f"🎯 **Tool to execute:** {tool_call_data.get('name')}\n"
                                        f"⚙️ **Arguments:** {json.dumps(tool_call_data.get('arguments', {}), indent=2)}",
                                        title="🔧 TOOL CALL DETAILS",
                                        style="bold cyan",
                                        border_style="cyan"
                                    ))
                                    
                                    # Find which server this tool belongs to
                                    server_id = None
                                    logger.debug("🔍 Finding server for tool...")
                                    server_actions = await llm_server.get_mcp_server_actions(['playwright', 'filesystem', 'sequential-thinking'])
                                    logger.debug(f"📋 Available servers: {list(server_actions.keys())}")
                                    
                                    for sid, actions in server_actions.items():
                                        logger.debug(f"🔍 Checking server {sid} with {len(actions)} actions")
                                        for action in actions:
                                            if action.get('name') == tool_call_data.get('name'):
                                                server_id = sid
                                                logger.info(f"✅ Found tool {tool_call_data.get('name')} on server {sid}")
                                                break
                                        if server_id:
                                            break
                                
                                    if server_id:
                                        console.print(Panel(
                                            f"🚀 Executing tool on server: {server_id}",
                                            title="⚡ TOOL EXECUTION STARTING",
                                            style="bold blue",
                                            border_style="blue"
                                        ))
                                        
                                        # Execute the tool
                                        execution_result = await llm_server.execute_mcp_action(
                                            server_id, 
                                            tool_call_data.get('name'), 
                                            tool_call_data.get('arguments', {})
                                        )
                                        
                                        logger.info(f"✅ Tool execution completed with result: {execution_result}")
                                        
                                        # Create structured tool response for user message
                                        tool_response_content = json.dumps(execution_result, indent=2)
                                        structured_response = f"<tool_response>\n{tool_response_content}\n</tool_response>"
                                        
                                        # Add structured tool response as user message
                                        llm_server.task_manager.add_message(task_id, 'user', structured_response)
                                        logger.debug(f"💾 Added structured tool response as user message: {len(structured_response)} characters")
                                        tool_call_executed = True
                                        
                                        console.print(Panel(
                                            f"🛠️ **Tool executed after approval**\n\n"
                                            f"**Result:**\n```json\n{tool_response_content}\n```\n\n"
                                            f"**Saved as user message:** `<tool_response>...</tool_response>`",
                                            title="⚡ TOOL EXECUTION COMPLETED",
                                            style="bold blue",
                                            border_style="blue"
                                        ))
                                    else:
                                        logger.error(f"❌ Could not find server for tool: {tool_call_data.get('name')}")
                                        console.print(Panel(
                                            f"❌ Could not find server for tool: {tool_call_data.get('name')}\n"
                                            f"📋 Available servers: {list(server_actions.keys())}\n"
                                            f"🔍 Tool name: {tool_call_data.get('name')}",
                                            title="🚨 SERVER NOT FOUND",
                                            style="bold red",
                                            border_style="red"
                                        ))
                                        
                                except Exception as e:
                                    logger.error(f"❌ Error executing approved tool: {str(e)}")
                                    import traceback
                                    logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                                    
                                    console.print(Panel(
                                        f"❌ Error executing approved tool: {str(e)}\n\n"
                                        f"**Full error details:**\n{traceback.format_exc()}",
                                        title="🚨 TOOL EXECUTION ERROR",
                                        style="bold red",
                                        border_style="red"
                                    ))
                            
                            # 3. Continue the task if tool was executed
                            if tool_call_executed:
                                # Trigger continuation of the conversation
                                await llm_server.continue_task_conversation(websocket, task_id)
                        
                        elif feedback_type == 'correct':
                            # Only show correction modal - actual correction handled by separate endpoint
                            pass
                        
                        elif feedback_type == 'comment':
                            # Add comment to chat history under user role (no tool execution)
                            if user_comment:
                                llm_server.task_manager.add_message(task_id, 'user', user_comment)
                                
                                console.print(Panel(
                                    f"💬 **User comment added to chat history**\n\n"
                                    f"**Comment:** {user_comment}\n\n"
                                    f"**Note:** Tool call was NOT executed - continuing with comment",
                                    title="📝 COMMENT ADDED",
                                    style="bold purple",
                                    border_style="purple"
                                ))
                                
                                # Continue the task after comment (no tool execution)
                                await llm_server.continue_task_conversation(websocket, task_id)
                        
                        # Send confirmation
                        await websocket.send(json.dumps({
                            'type': 'learning_feedback_processed',
                            'feedback_type': feedback_type,
                            'success': True
                        }))
                        
                    except Exception as e:
                        logger.error(f"Error processing learning feedback: {e}")
                        import traceback
                        traceback.print_exc()
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'error': f'Failed to process learning feedback: {str(e)}'
                        }))
                    
                    continue
                
                # Handle chat requests (existing OpenAI-format)
                messages = request.get('messages', [])
                model = request.get('model', 'qwen3')
                stream = request.get('stream', True)
                temperature = request.get('temperature', 0.7)
                max_tokens = request.get('max_tokens', 2048)
                task_id = request.get('task_id')  # Optional task ID for persistence
                auto_response = request.get('auto_response', False)  # Flag for auto-responses
                
                logger.info(f"🐛 Main handler received request: auto_response={auto_response}, task_id={task_id}")

                if not messages:
                    error_response = {
                        "error": {
                            "message": "No messages provided",
                            "type": "invalid_request"
                        }
                    }
                    await websocket.send(json.dumps(error_response))
                    continue

                # Save user message to task if task_id provided and not an auto-response
                if task_id and messages and not auto_response:
                    latest_message = messages[-1]  # Get the most recent message
                    if latest_message.get('role') == 'user':
                        llm_server.task_manager.add_message(task_id, 'user', latest_message.get('content', ''))
                        logger.info(f"Saved user message to task {task_id}")
                elif auto_response:
                    logger.info(f"Auto-response request for task {task_id} - not saving user message")

                if not stream:
                    # Non-streaming response (fallback)
                    await llm_server.initialize_model(model)
                    prompt = llm_server.messages_to_prompt(messages)
                    
                    response_text = await llm_server.llm.generate(prompt)
                    
                    response = {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }]
                    }
                    await websocket.send(json.dumps(response))
                else:
                    # Streaming response
                    if auto_response:
                        logger.info(f"🐛 Main handler: AUTO-RESPONSE for task {task_id}")
                    else:
                        logger.info(f"🐛 Main handler: USER-INITIATED request for task {task_id}")
                    
                    await llm_server.stream_response(
                        websocket, 
                        messages, 
                        model, 
                        temperature, 
                        max_tokens,
                        task_id
                    )
                
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                error_response = {
                    "error": {
                        "message": "Invalid JSON format",
                        "type": "parse_error"
                    }
                }
                await websocket.send(json.dumps(error_response))
            
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                import traceback
                traceback.print_exc()
                
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "server_error"
                    }
                }
                await websocket.send(json.dumps(error_response))
    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_addr}")
    except Exception as e:
        logger.error(f"Connection error: {e}")
    finally:
        # Clean up MCP connections when client disconnects
        await llm_server.cleanup_mcp_connections()

async def main():
    """Start the WebSocket server"""
    host = "localhost"
    port = 8000
    
    console.print(Panel(
        f"🚀 **Dark.RL WebSocket Server**\n\n"
        f"🌐 **Host:** {host}\n"
        f"🔌 **Port:** {port}\n"
        f"🤖 **Engine:** AsyncOnlineLLM\n"
        f"🛠️ **Features:** Real MCP servers, API key management, tool calling\n\n"
        f"**Connect your frontend to:** ws://{host}:{port}\n\n"
        f"⚡ **Ready for connections!**",
        title="🔥 DARK.RL SERVER STARTING",
        style="bold green",
        border_style="green"
    ))
    
    # Initialize global LLM server and load model during startup
    global global_llm_server
    console.print(Panel(
        "🧠 Initializing AI model...\n"
        "This may take a moment on first startup",
        title="🔄 MODEL LOADING",
        style="bold yellow",
        border_style="yellow"
    ))
    
    global_llm_server = DarkRLLLMServer()
    try:
        await global_llm_server.initialize_model("qwen2.5-vl")  # Load default model
        console.print(Panel(
            "✅ AI model loaded successfully!\n"
            "🎯 Ready for immediate responses",
            title="🧠 MODEL READY",
            style="bold green",
            border_style="green"
        ))
    except Exception as e:
        console.print(Panel(
            f"❌ Failed to load model: {str(e)}\n"
            "⚠️ Model will be loaded on first request",
            title="🚨 MODEL LOADING ERROR",
            style="bold red",
            border_style="red"
        ))
        # Continue with server startup even if model loading fails
    
    # Start server on root path
    async with websockets.serve(handle_client, host, port):
        console.print(Panel(
            "✅ Server started successfully!\n"
            "🎯 Ready to handle WebSocket connections",
            title="🚀 SERVER READY",
            style="bold blue",
            border_style="blue"
        ))
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print(Panel(
            "🛑 Server stopped by user",
            title="👋 SHUTDOWN",
            style="bold yellow",
            border_style="yellow"
        ))
        # Clean up global resources
        if global_llm_server is not None:
            try:
                # Cleanup MCP connections
                asyncio.run(global_llm_server.cleanup_mcp_connections())
            except:
                pass
    except Exception as e:
        console.print(Panel(
            f"❌ Server error: {str(e)}",
            title="🚨 SERVER ERROR",
            style="bold red",
            border_style="red"
        ))
        import traceback
        traceback.print_exc() 