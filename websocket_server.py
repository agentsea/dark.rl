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
import json_repair
from typing import Dict, Any, List
from pathlib import Path

# Import Dark.RL components
from dark.online_llm import AsyncOnlineLLM, BatchConfig
from dark.sampling_params import SamplingParams
from task_manager import TaskManager
from mcp_use import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            "custom": "Qwen/Qwen3-8B"  # fallback
        }
        self.current_model = None
        self.task_manager = TaskManager()
        
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
                "requires_api_key": False
            },
            {
                "id": "filesystem",
                "name": "File System",
                "description": "File and directory operations",
                "config": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/ubuntu/dark.rl"]
                },
                "requires_api_key": False
            },
            {
                "id": "sequential-thinking",
                "name": "Sequential Thinking",
                "description": "Sequential reasoning and thought processes",
                "config": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
                },
                "requires_api_key": False
            },
            {
                "id": "firecrawl",
                "name": "Firecrawl",
                "description": "Web scraping and content extraction",
                "config": {
                    "command": "npx",
                    "args": ["-y", "firecrawl-mcp"],
                    "env": {
                        "FIRECRAWL_API_KEY": ""  # Will be populated dynamically
                    }
                },
                "requires_api_key": True,
                "api_key_env": "FIRECRAWL_API_KEY"
            },
            {
                "id": "postgres",
                "name": "PostgreSQL",
                "description": "Database operations",
                "config": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-postgres"],
                    "env": {
                        "POSTGRES_CONNECTION_STRING": ""  # Will be populated dynamically
                    }
                },
                "requires_api_key": True,
                "api_key_env": "POSTGRES_CONNECTION_STRING"
            },
            {
                "id": "github",
                "name": "GitHub",
                "description": "Git repository management",
                "config": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {
                        "GITHUB_PERSONAL_ACCESS_TOKEN": ""  # Will be populated dynamically
                    }
                },
                "requires_api_key": True,
                "api_key_env": "GITHUB_PERSONAL_ACCESS_TOKEN"
            },
            {
                "id": "slack",
                "name": "Slack",
                "description": "Team communication",
                "config": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-slack"],
                    "env": {
                        "SLACK_BOT_TOKEN": ""  # Will be populated dynamically
                    }
                },
                "requires_api_key": True,
                "api_key_env": "SLACK_BOT_TOKEN"
            }
        ]
        
        # MCP client connections cache
        self.mcp_clients = {}
        self.mcp_sessions = {}
        
        # MCP server actions cache
        self.mcp_actions_cache = {}
    
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
            # Check if API key is configured if required
            api_key_available = True
            if server.get("requires_api_key", False):
                api_key_env = server.get("api_key_env", "")
                api_key_available = bool(get_api_key(api_key_env))
            
            server_info = {
                "id": server["id"],
                "name": server["name"],
                "description": server["description"],
                "requires_api_key": server.get("requires_api_key", False),
                "api_key_available": api_key_available,
                "api_key_env": server.get("api_key_env", "") if server.get("requires_api_key", False) else None
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
        
        # Check API key if required
        if server_config.get("requires_api_key", False):
            api_key_env = server_config.get("api_key_env", "")
            if not get_api_key(api_key_env):
                raise ValueError(f"API key {api_key_env} not configured for server {server_id}")
        
        # Create MCP config with dynamic API key population
        config = server_config["config"].copy()
        
        # Populate API keys dynamically
        if "env" in config:
            for env_var, _ in config["env"].items():
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
                    # Build parameter information from inputSchema
                    parameters = {}
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        schema = tool.inputSchema
                        if isinstance(schema, dict) and 'properties' in schema:
                            required = schema.get('required', [])
                            for param_name, param_info in schema['properties'].items():
                                param_type = param_info.get('type', 'string')
                                param_desc = param_info.get('description', '')
                                parameters[param_name] = {
                                    "type": param_type,
                                    "description": param_desc,
                                    "required": param_name in required
                                }
                    
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
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
                
                # Convert parameters to proper schema format
                for param_name, param_info in parameters.items():
                    tool_schema["parameters"]["properties"][param_name] = {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", "")
                    }
                    
                    if param_info.get("required", False):
                        tool_schema["parameters"]["required"].append(param_name)
                
                tools.append(tool_schema)
        
        return tools

    def build_qwen_chat_messages(self, messages: List[dict], tools: List[dict] = None) -> List[dict]:
        """Build messages in Qwen chat template format with tools"""
        chat_messages = []
        
        # Build system message with tools
        system_content = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        
        if tools:
            system_content += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            system_content += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            
            for tool in tools:
                system_content += json.dumps(tool, indent=2) + "\n"
            
            system_content += "</tools>\n\n"
            system_content += 'For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:\n'
            system_content += '<tool_call>\n{"name": "function_name", "arguments": {"param": "value"}}\n</tool_call>\n\n'
            system_content += 'You can also include your reasoning in <think></think> tags before making tool calls.'
        
        chat_messages.append({"role": "system", "content": system_content})
        
        # Add conversation messages
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role in ['user', 'assistant', 'system']:
                chat_messages.append({"role": role, "content": content})
        
        return chat_messages

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

    async def should_use_action_mode(self, messages: List[dict]) -> tuple[bool, List[str]]:
        """Determine if we should use action mode based on MCP server mentions"""
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

    async def execute_mcp_action(self, server_id: str, action_name: str, parameters: dict) -> dict:
        """Execute an MCP action on a server"""
        try:
            session = await self.get_mcp_session(server_id)
            result = await session.connector.call_tool(action_name, parameters)
            
            # Convert result to dict format
            return {
                "success": True,
                "result": result.model_dump() if hasattr(result, 'model_dump') else str(result),
                "content": result.content if hasattr(result, 'content') else str(result)
            }
        except Exception as e:
            logger.error(f"Error executing MCP action {action_name} on server {server_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": f"Error executing {action_name}: {str(e)}"
            }

    def create_task(self, initial_prompt: str, model: str = "qwen2.5-vl") -> str:
        """Create a new task and return its ID"""
        return self.task_manager.create_task(initial_prompt, model)
    
    def get_task(self, task_id: str) -> dict:
        """Get task details by ID"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return {"error": "Task not found"}
        
        messages = self.task_manager.get_task_messages(task_id)
        return {
            "task": task,
            "messages": messages
        }
    
    def get_recent_tasks(self) -> list:
        """Get list of recent tasks"""
        return self.task_manager.get_recent_tasks()

    async def stream_response(self, websocket, messages: list, model: str = "qwen3", temperature: float = 0.7, max_tokens: int = 2048, task_id: str = None):
        """Stream response from the actual LLM with MCP action support"""
        
        try:
            # Initialize model if needed
            await self.initialize_model(model)
            
            # Check if we should use action mode based on MCP server mentions
            use_action_mode, mentioned_servers = await self.should_use_action_mode(messages)
            
            if use_action_mode:
                logger.info(f"🎯 Action mode detected! Mentioned servers: {mentioned_servers}")
                await self.stream_action_response(websocket, messages, mentioned_servers, model, temperature, max_tokens, task_id)
            else:
                logger.info("💬 Standard chat mode")
                await self.stream_chat_response(websocket, messages, model, temperature, max_tokens, task_id)
                
        except Exception as e:
            logger.error(f"Error during streaming setup: {e}")
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
        
        logger.info(f"Streaming chat response for {len(chat_messages)} messages...")
        
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
                    logger.info(f"Saved assistant response to task {task_id}")
            else:
                logger.info(f"Streaming stopped due to closed connection. Sent {chunk_count} chunks")
                
        except Exception as stream_error:
            logger.error(f"Error in streaming loop: {stream_error}")
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
            tools = await self.build_tools_schema(mentioned_servers)
            
            if not tools:
                # No tools found, fall back to chat mode
                logger.warning(f"No tools found for servers: {mentioned_servers}")
                await self.stream_chat_response(websocket, messages, model, temperature, max_tokens, task_id)
                return
            
            # Build Qwen chat messages with tools
            chat_messages = self.build_qwen_chat_messages(messages, tools)
            
            logger.info(f"🛠️ Built Qwen chat with {len(tools)} tools")
            
            # Create sampling parameters optimized for tool calling
            sampling_params = SamplingParams(
                temperature=max(0.1, min(temperature, 0.4)),  # Moderate temperature for reasoning + tool calls
                max_tokens=min(max_tokens, 800),  # More tokens for thinking + tool calls
                n=1,
                presence_penalty=0.1,
                ignore_eos=False
            )
            
            # Stream the response using chat format
            accumulated_response = ""
            chunk_count = 0
            
            # Use chat_stream if available
            if hasattr(self.llm, 'chat_stream'):
                import inspect
                sig = inspect.signature(self.llm.chat_stream)
                if 'sampling_params' in sig.parameters:
                    stream_iterator = self.llm.chat_stream(chat_messages, sampling_params=sampling_params)
                else:
                    stream_iterator = self.llm.chat_stream(chat_messages)
            else:
                # Fallback to prompt-based streaming
                prompt = self.llm._messages_to_prompt(chat_messages)
                stream_iterator = self.llm.stream(prompt, sampling_params=sampling_params)
            
            # Handle streaming
            if hasattr(stream_iterator, '__aiter__'):
                # Async generator
                async for chunk in stream_iterator:
                    if websocket.close_code is not None:
                        logger.warning("WebSocket closed during action streaming")
                        break
                    
                    accumulated_response += chunk
                    chunk_count += 1
                    
                    # Check if we hit a tool call that needs user approval
                    if ("<tool_call>" in accumulated_response and 
                        "</tool_call>" in accumulated_response and
                        chunk_count > 5):  # Make sure we have substantial content
                        
                        # Pause streaming and wait for user approval
                        logger.info("🛠️ Tool call detected, pausing for user approval")
                        
                        # Stream current content
                        stream_chunk = {
                            "choices": [{
                                "delta": {
                                    "content": chunk
                                },
                                "finish_reason": "tool_call_requested"
                            }]
                        }
                        
                        try:
                            await websocket.send(json.dumps(stream_chunk))
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed while sending tool call chunk")
                            break
                        
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
                        logger.warning("WebSocket connection closed while sending action chunk")
                        break
            else:
                # Regular generator
                for chunk in stream_iterator:
                    if websocket.close_code is not None:
                        logger.warning("WebSocket closed during action streaming")
                        break
                    
                    accumulated_response += chunk
                    chunk_count += 1
                    
                    # Check if we hit a tool call that needs user approval
                    if ("<tool_call>" in accumulated_response and 
                        "</tool_call>" in accumulated_response and
                        chunk_count > 5):  # Make sure we have substantial content
                        
                        # Pause streaming and wait for user approval
                        logger.info("🛠️ Tool call detected, pausing for user approval")
                        
                        # Stream current content
                        stream_chunk = {
                            "choices": [{
                                "delta": {
                                    "content": chunk
                                },
                                "finish_reason": "tool_call_requested"
                            }]
                        }
                        
                        try:
                            await websocket.send(json.dumps(stream_chunk))
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed while sending tool call chunk")
                            break
                        
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
                        logger.warning("WebSocket connection closed while sending action chunk")
                        break
            
            # If we reach here, no tool call was detected, send final chunk
            if websocket.close_code is None:
                final_chunk = {
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                
                await websocket.send(json.dumps(final_chunk))
                logger.info(f"Tool-enabled response completed. Length: {len(accumulated_response)} characters")
                
                # Save to task if provided
                if task_id and accumulated_response:
                    self.task_manager.add_message(task_id, 'assistant', accumulated_response)
                    logger.info(f"Saved tool response to task {task_id}")
            
        except Exception as e:
            logger.error(f"Error in action streaming: {e}")
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

async def handle_client(websocket):
    """Handle WebSocket client connections"""
    client_addr = websocket.remote_address
    logger.info(f"Client connected: {client_addr}")
    
    llm_server = DarkRLLLMServer()
    
    try:
        async for message in websocket:
            try:
                # Parse request
                request = json.loads(message)
                logger.info(f"Received request from {client_addr}")
                
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
                    
                    logger.info(f"🚀 Executing tool {tool_name} on server {server_id} and continuing stream")
                    
                    try:
                        # Execute the tool
                        action_result = await llm_server.execute_mcp_action(server_id, tool_name, parameters)
                        
                        # Format tool response in Qwen format
                        tool_response_content = f"\n<tool_response>\n{json.dumps(action_result, indent=2)}\n</tool_response>\n\n"
                        
                        # Stream the tool response
                        tool_response_chunk = {
                            "choices": [{
                                "delta": {
                                    "content": tool_response_content
                                },
                                "finish_reason": None
                            }]
                        }
                        
                        await websocket.send(json.dumps(tool_response_chunk))
                        
                        # Now continue generating the assistant's response after the tool call
                        # Build context with the tool response
                        updated_response = current_response + tool_response_content
                        
                        # Create a mock message to continue generation
                        continue_messages = [
                            {"role": "assistant", "content": updated_response}
                        ]
                        
                        # Build tools schema and chat messages for continuation
                        tools = await llm_server.build_tools_schema(mentioned_servers)
                        chat_messages = llm_server.build_qwen_chat_messages([], tools)  # Empty conversation for context
                        
                        # Add the partial assistant response to continue from
                        chat_messages.append({"role": "assistant", "content": updated_response})
                        
                        # Create sampling parameters for continuation
                        sampling_params = SamplingParams(
                            temperature=0.3,
                            max_tokens=400,  # Remaining tokens for conclusion
                            n=1,
                            presence_penalty=0.1,
                            ignore_eos=False
                        )
                        
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
                            
                            # Save complete response to task
                            if task_id:
                                complete_response = updated_response + continuation_response
                                llm_server.task_manager.add_message(task_id, 'assistant', complete_response)
                                logger.info(f"Saved complete tool response to task {task_id}")
                    
                    except Exception as e:
                        logger.error(f"Error executing tool and continuing: {e}")
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

                # Handle learning feedback requests
                if request.get('type') == 'learning_feedback':
                    logger.info(f"Received learning_feedback request: {request}")
                    
                    feedback_type = request.get('feedback_type')
                    message = request.get('message')
                    task_id = request.get('task_id')
                    user_comment = request.get('user_comment')
                    
                    logger.info(f"Learning feedback: {feedback_type} for task {task_id}")
                    
                    try:
                        # Initialize model if needed
                        await llm_server.initialize_model()
                        
                        # Get task and message history for context
                        task_data = llm_server.get_task(task_id)
                        if task_data.get('error'):
                            await websocket.send(json.dumps({
                                'type': 'error',
                                'error': 'Task not found'
                            }))
                            continue
                        
                        # Get all messages for this task to build context
                        messages = task_data.get('messages', [])
                        
                        # Build conversation context
                        conversation_context = []
                        for msg in messages:
                            conversation_context.append({
                                "role": msg['role'],
                                "content": msg['content']
                            })
                        
                        # Apply learning based on feedback type (following train_one_vl.py logic)
                        if feedback_type == 'approve':
                            # Positive reinforcement learning
                            logger.info("Learning from approved response...")
                            
                            # Find the user message that led to this assistant response
                            user_prompt = None
                            for i, msg in enumerate(conversation_context):
                                if msg['role'] == 'assistant' and msg['content'] == message['content']:
                                    # Find the previous user message
                                    for j in range(i-1, -1, -1):
                                        if conversation_context[j]['role'] == 'user':
                                            user_prompt = conversation_context[j]['content']
                                            break
                                    break
                            
                            if user_prompt:
                                # Create learning example
                                learning_example = [
                                    {"role": "user", "content": user_prompt},
                                    {"role": "assistant", "content": message['content']}
                                ]
                                
                                # Apply positive learning (from train_one_vl.py: OPTIMAL_LEARNING_STEPS=10, OPTIMAL_LEARNING_RATE=1e-4)
                                await llm_server.llm.learn(
                                    learning_example,
                                    adapter="task_learning",
                                    steps=10,
                                    lr=1e-4
                                )
                                
                                logger.info("✓ Learning completed for approved response")
                        
                        elif feedback_type == 'deny':
                            # Negative learning with user feedback
                            logger.info("Learning from denied response...")
                            
                            if user_comment:
                                # Create critique context (following train_one_vl.py critique_context)
                                critique_context = f"You are tasked with determining if an action taken by an agent to accomplish a task is correct. The task was: {task_data['task']['title']}. The response was: {message['content']}. The user feedback was: {user_comment}. Please output a critique of the response."
                                
                                # Apply negative learning
                                await llm_server.llm.learn(
                                    [{"role": "user", "content": critique_context}, {"role": "assistant", "content": user_comment}],
                                    adapter="critique_learning",
                                    steps=10,
                                    lr=1e-4
                                )
                                
                                logger.info("✓ Learning completed for denied response")
                        
                        elif feedback_type == 'correct':
                            # Learning with correction
                            logger.info("Learning from corrected response...")
                            
                            if user_comment:
                                # Find the user message that led to this assistant response
                                user_prompt = None
                                for i, msg in enumerate(conversation_context):
                                    if msg['role'] == 'assistant' and msg['content'] == message['content']:
                                        # Find the previous user message
                                        for j in range(i-1, -1, -1):
                                            if conversation_context[j]['role'] == 'user':
                                                user_prompt = conversation_context[j]['content']
                                                break
                                        break
                                
                                if user_prompt:
                                    # Create corrected learning example
                                    corrected_example = [
                                        {"role": "user", "content": user_prompt},
                                        {"role": "assistant", "content": user_comment}  # Use user's correction as the right answer
                                    ]
                                    
                                    # Apply corrective learning
                                    await llm_server.llm.learn(
                                        corrected_example,
                                        adapter="task_learning",
                                        steps=10,
                                        lr=1e-4
                                    )
                                    
                                    logger.info("✓ Learning completed for corrected response")
                        
                        elif feedback_type == 'comment':
                            # General feedback/comment
                            logger.info("Processing general feedback...")
                            
                            if user_comment:
                                # Store feedback for future reference
                                feedback_context = f"Task: {task_data['task']['title']}. Response: {message['content']}. User feedback: {user_comment}. Consider this feedback for future responses."
                                
                                # Light learning from general feedback
                                await llm_server.llm.learn(
                                    [{"role": "user", "content": feedback_context}, {"role": "assistant", "content": "I understand and will consider this feedback."}],
                                    adapter="general_feedback",
                                    steps=5,  # Less intensive for general feedback
                                    lr=5e-5   # Lower learning rate for general feedback
                                )
                                
                                logger.info("✓ General feedback processed")
                        
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
    
    logger.info(f"Starting Dark.RL WebSocket server on {host}:{port}")
    logger.info("Using real AsyncOnlineLLM for responses")
    logger.info("Connect your frontend to: ws://localhost:8000")
    
    # Start server on root path
    async with websockets.serve(handle_client, host, port):
        logger.info("Server started successfully")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc() 