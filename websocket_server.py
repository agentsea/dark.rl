#!/usr/bin/env python3
"""
WebSocket server for Dark.RL frontend using real AsyncOnlineLLM.
Accepts OpenAI-format requests and returns streaming responses from actual models.
Now includes task management with SQLite persistence.
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, Any

# Import Dark.RL components
from dark.online_llm import AsyncOnlineLLM, BatchConfig
from dark.sampling_params import SamplingParams
from task_manager import TaskManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def get_mcp_server_actions(self, server_ids: list) -> dict:
        """Get available actions for specified MCP servers"""
        server_actions = {}
        
        # Mock actions for different server types
        # In real implementation, this would query the actual MCP servers
        mock_actions = {
            "playwright": [
                {"name": "navigate", "description": "Navigate to a web page", "parameters": {"url": "string"}},
                {"name": "click", "description": "Click an element", "parameters": {"selector": "string"}},
                {"name": "fill", "description": "Fill form input", "parameters": {"selector": "string", "value": "string"}},
                {"name": "screenshot", "description": "Take a screenshot", "parameters": {"path": "string"}},
                {"name": "wait_for_element", "description": "Wait for element to appear", "parameters": {"selector": "string"}}
            ],
            "filesystem": [
                {"name": "read_file", "description": "Read contents of a file", "parameters": {"path": "string"}},
                {"name": "write_file", "description": "Write content to a file", "parameters": {"path": "string", "content": "string"}},
                {"name": "list_directory", "description": "List directory contents", "parameters": {"path": "string"}},
                {"name": "create_directory", "description": "Create a new directory", "parameters": {"path": "string"}},
                {"name": "delete_file", "description": "Delete a file", "parameters": {"path": "string"}}
            ],
            "github": [
                {"name": "create_repo", "description": "Create a new repository", "parameters": {"name": "string", "description": "string"}},
                {"name": "list_repos", "description": "List user repositories", "parameters": {}},
                {"name": "create_issue", "description": "Create a new issue", "parameters": {"title": "string", "body": "string"}},
                {"name": "list_issues", "description": "List repository issues", "parameters": {"repo": "string"}},
                {"name": "create_pull_request", "description": "Create a pull request", "parameters": {"title": "string", "body": "string", "head": "string", "base": "string"}}
            ],
            "postgres": [
                {"name": "execute_query", "description": "Execute SQL query", "parameters": {"query": "string"}},
                {"name": "create_table", "description": "Create a new table", "parameters": {"table_name": "string", "schema": "string"}},
                {"name": "insert_data", "description": "Insert data into table", "parameters": {"table_name": "string", "data": "object"}},
                {"name": "select_data", "description": "Select data from table", "parameters": {"table_name": "string", "where": "string"}}
            ],
            "slack": [
                {"name": "send_message", "description": "Send message to channel", "parameters": {"channel": "string", "message": "string"}},
                {"name": "list_channels", "description": "List available channels", "parameters": {}},
                {"name": "create_channel", "description": "Create a new channel", "parameters": {"name": "string", "description": "string"}},
                {"name": "upload_file", "description": "Upload file to channel", "parameters": {"channel": "string", "file_path": "string"}}
            ]
        }
        
        for server_id in server_ids:
            if server_id in mock_actions:
                server_actions[server_id] = mock_actions[server_id]
            else:
                # Default actions for unknown servers
                server_actions[server_id] = [
                    {"name": "generic_action", "description": f"Generic action for {server_id}", "parameters": {"input": "string"}}
                ]
        
        return server_actions

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
        """Stream response from the actual LLM"""
        
        try:
            # Initialize model if needed
            await self.initialize_model(model)
            
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
            
            logger.info(f"Streaming response for {len(chat_messages)} messages...")
            
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

                # Handle MCP server actions requests
                if request.get('type') == 'get_mcp_server_actions':
                    server_ids = request.get('server_ids', [])
                    server_actions = llm_server.get_mcp_server_actions(server_ids)
                    
                    # Send response for each server
                    for server_id, actions in server_actions.items():
                        response = {
                            "type": "mcp_server_actions_response",
                            "server_id": server_id,
                            "actions": actions
                        }
                        await websocket.send(json.dumps(response))
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