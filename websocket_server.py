#!/usr/bin/env python3
"""
WebSocket server for Dark.RL frontend using real AsyncOnlineLLM.
Accepts OpenAI-format requests and returns streaming responses from actual models.
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, Any

# Import Dark.RL components
from dark.online_llm import AsyncOnlineLLM, BatchConfig
from dark.sampling_params import SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DarkRLLLMServer:
    """Dark.RL LLM server that provides real AI responses"""
    
    def __init__(self):
        self.llm = None
        self.models = {
            "qwen3": "Qwen/Qwen3-8B",
            "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct", 
            "custom": "Qwen/Qwen3-8B"  # fallback
        }
        self.current_model = None
        
        # Mock MCP servers for now - in real implementation these would come from MCP registry
        self.mcp_servers = [
            {"id": "playwright", "name": "Playwright", "description": "Web automation and testing"},
            {"id": "postgres", "name": "PostgreSQL", "description": "Database operations"},
            {"id": "filesystem", "name": "File System", "description": "File and directory operations"},
            {"id": "github", "name": "GitHub", "description": "Git repository management"},
            {"id": "slack", "name": "Slack", "description": "Team communication"},
            {"id": "jira", "name": "Jira", "description": "Project management and issue tracking"},
            {"id": "docker", "name": "Docker", "description": "Container management"},
            {"id": "aws", "name": "AWS", "description": "Amazon Web Services"},
            {"id": "plausible", "name": "Plausible Analytics", "description": "Website analytics"},
            {"id": "prometheus", "name": "Prometheus", "description": "Monitoring and alerting"}
        ]
    
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
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def get_mcp_servers(self, query: str = "") -> list:
        """Get list of MCP servers, optionally filtered by query"""
        if not query:
            return self.mcp_servers
        
        query = query.lower()
        filtered = []
        
        for server in self.mcp_servers:
            # Search in name and description
            if (query in server["name"].lower() or 
                query in server["description"].lower() or
                query in server["id"].lower()):
                filtered.append(server)
        
        return filtered

    async def stream_response(self, websocket, messages: list, model: str = "qwen3", temperature: float = 0.7, max_tokens: int = 2048):
        """Stream response from the actual LLM"""
        
        try:
            # Initialize model if needed
            await self.initialize_model(model)
            
            # Convert messages to prompt
            prompt = self.messages_to_prompt(messages)
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                presence_penalty=0.1,
                ignore_eos=False
            )
            
            logger.info(f"Streaming response for prompt: {prompt[:100]}...")
            
            # Stream response from actual LLM
            accumulated_response = ""
            chunk_count = 0
            
            try:
                async for chunk in self.llm.stream(prompt, sampling_params=sampling_params):
                    # Check if websocket is still open
                    if websocket.close_code is not None:
                        logger.warning("WebSocket closed during streaming")
                        break
                    
                    accumulated_response += chunk
                    chunk_count += 1
                    
                    logger.debug(f"Chunk {chunk_count}: {repr(chunk[:50])}")
                    
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
                
                # Handle chat requests (existing OpenAI-format)
                messages = request.get('messages', [])
                model = request.get('model', 'qwen3')
                stream = request.get('stream', True)
                temperature = request.get('temperature', 0.7)
                max_tokens = request.get('max_tokens', 2048)
                
                if not messages:
                    error_response = {
                        "error": {
                            "message": "No messages provided",
                            "type": "invalid_request"
                        }
                    }
                    await websocket.send(json.dumps(error_response))
                    continue
                
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
                        max_tokens
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