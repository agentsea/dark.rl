#!/usr/bin/env python3
"""
Task management system for Dark.RL with SQLite storage.
Handles task creation, message history, and persistence.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

class TaskManager:
    """Manages tasks and conversation history in SQLite"""
    
    def __init__(self, db_path: str = "dark_rl_tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    initial_prompt TEXT NOT NULL,
                    model TEXT DEFAULT 'qwen2.5-vl',
                    mcp_servers TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    current_state TEXT DEFAULT 'idle'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks (id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_task_id 
                ON messages (task_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages (timestamp)
            """)
            
            # New table for storing response preferences
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    local_response TEXT NOT NULL,
                    gpt_response TEXT NOT NULL,
                    preferred_model TEXT NOT NULL,  -- 'local' or 'gpt'
                    local_model_name TEXT,
                    gpt_model_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks (id)
                )
            """)

            # New table for storing pending dual responses
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_dual_responses (
                    task_id TEXT PRIMARY KEY,
                    local_response TEXT NOT NULL,
                    gpt_response TEXT NOT NULL,
                    local_model TEXT,
                    gpt_model TEXT,
                    local_finished BOOLEAN DEFAULT 0,
                    gpt_finished BOOLEAN DEFAULT 0,
                    session_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks (id)
                )
            """)
            
            conn.commit()
    
    def create_task(self, initial_prompt: str, model: str = "qwen2.5-vl") -> str:
        """Create a new task and return its ID"""
        task_id = str(uuid.uuid4())
        
        # Generate a title from the first few words of the prompt
        words = initial_prompt.split()[:8]
        title = " ".join(words) + ("..." if len(initial_prompt.split()) > 8 else "")
        
        # Extract MCP servers referenced with @ symbol
        mcp_servers = self._extract_mcp_servers(initial_prompt)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO tasks (id, title, initial_prompt, model, mcp_servers)
                VALUES (?, ?, ?, ?, ?)
            """, (task_id, title, initial_prompt, model, json.dumps(mcp_servers)))
            
            # Add the initial user message
            conn.execute("""
                INSERT INTO messages (task_id, role, content)
                VALUES (?, ?, ?)
            """, (task_id, 'user', initial_prompt))
            
            conn.commit()
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task details by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM tasks WHERE id = ?
            """, (task_id,))
            
            row = cursor.fetchone()
            if row:
                task_data = dict(row)
                # Parse MCP servers JSON string back to list
                try:
                    task_data['mcp_servers'] = json.loads(task_data['mcp_servers'])
                except (json.JSONDecodeError, TypeError):
                    task_data['mcp_servers'] = []
                return task_data
            return None
    
    def get_task_messages(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a task"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT role, content, timestamp 
                FROM messages 
                WHERE task_id = ? 
                ORDER BY timestamp ASC
            """, (task_id,))
            
            return [dict(row) for row in cursor.fetchall()]

    def get_task_messages_with_indexes(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a task with their indexes for correction purposes"""
        messages = self.get_task_messages(task_id)
        
        # Add index to each message
        for i, message in enumerate(messages):
            message['index'] = i
            
        return messages
    
    def add_message(self, task_id: str, role: str, content: str) -> bool:
        """Add a message to a task"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO messages (task_id, role, content)
                    VALUES (?, ?, ?)
                """, (task_id, role, content))
                
                # Update task's updated_at timestamp
                conn.execute("""
                    UPDATE tasks SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (task_id,))
                
                conn.commit()
            return True
        except Exception as e:
            print(f"Error adding message: {e}")
            return False

    def replace_message(self, task_id: str, message_index: int, new_content: str) -> bool:
        """Replace a message at the specified index with new content"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First get all messages ordered by timestamp to find the correct message ID
                cursor = conn.execute("""
                    SELECT id FROM messages 
                    WHERE task_id = ? 
                    ORDER BY timestamp ASC
                """, (task_id,))
                
                message_ids = [row[0] for row in cursor.fetchall()]
                
                # Check if the index is valid
                if message_index < 0 or message_index >= len(message_ids):
                    print(f"Error: Invalid message index {message_index}. Task has {len(message_ids)} messages.")
                    return False
                
                # Get the message ID at the specified index
                target_message_id = message_ids[message_index]
                
                # Update the message content
                conn.execute("""
                    UPDATE messages 
                    SET content = ?, timestamp = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_content, target_message_id))
                
                # Update task's updated_at timestamp
                conn.execute("""
                    UPDATE tasks SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (task_id,))
                
                conn.commit()
                
                print(f"✅ Successfully replaced message {message_index} in task {task_id}")
                return True
                
        except Exception as e:
            print(f"Error replacing message: {e}")
            return False
    
    def get_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tasks ordered by updated_at"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, title, initial_prompt, model, created_at, updated_at, status
                FROM tasks 
                ORDER BY updated_at DESC 
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, task_id))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error updating task status: {e}")
            return False
    
    def task_exists(self, task_id: str) -> bool:
        """Check if a task exists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 1 FROM tasks WHERE id = ? LIMIT 1
            """, (task_id,))
            return cursor.fetchone() is not None
    
    def _extract_mcp_servers(self, text: str) -> List[str]:
        """Extract MCP server references from text (e.g., @playwright, @github)"""
        import re
        
        # Find all @ mentions followed by word characters
        # Pattern: @ followed by word characters (letters, numbers, underscore, hyphen)
        pattern = r'@([a-zA-Z0-9_-]+)'
        matches = re.findall(pattern, text)
        
        # Remove duplicates and return
        return list(set(matches)) 

    def save_response_preference(self, task_id: str, user_prompt: str, local_response: str, 
                               gpt_response: str, preferred_model: str, local_model_name: str = None, 
                               gpt_model_name: str = None) -> int:
        """Save user's response preference for analytics and learning"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO response_preferences 
                (task_id, user_prompt, local_response, gpt_response, preferred_model, local_model_name, gpt_model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (task_id, user_prompt, local_response, gpt_response, preferred_model, local_model_name, gpt_model_name))
            
            conn.commit()
            return cursor.lastrowid

    def get_response_preferences(self, task_id: str = None) -> list:
        """Get response preferences, optionally filtered by task_id"""
        with sqlite3.connect(self.db_path) as conn:
            if task_id:
                cursor = conn.execute("""
                    SELECT * FROM response_preferences WHERE task_id = ?
                    ORDER BY created_at DESC
                """, (task_id,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM response_preferences
                    ORDER BY created_at DESC
                """)
            
            return [dict(row) for row in cursor.fetchall()]

    # New methods for simplified state management
    def get_task_state(self, task_id: str) -> str:
        """Get current task state (idle, streaming_single, streaming_dual, awaiting_dual_selection)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT current_state FROM tasks WHERE id = ?
            """, (task_id,))
            
            row = cursor.fetchone()
            return row[0] if row else 'idle'

    def set_task_state(self, task_id: str, state: str) -> bool:
        """Set current task state"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE tasks SET current_state = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (state, task_id))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error setting task state: {e}")
            return False

    def get_pending_dual_responses(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get pending dual responses for a task"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM pending_dual_responses WHERE task_id = ?
            """, (task_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def set_pending_dual_responses(self, task_id: str, local_response: str, gpt_response: str, 
                                 local_model: str = None, gpt_model: str = None, 
                                 local_finished: bool = False, gpt_finished: bool = False,
                                 session_id: int = None) -> bool:
        """Save or update pending dual responses"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pending_dual_responses 
                    (task_id, local_response, gpt_response, local_model, gpt_model, 
                     local_finished, gpt_finished, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (task_id, local_response, gpt_response, local_model, gpt_model, 
                      local_finished, gpt_finished, session_id))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error setting pending dual responses: {e}")
            return False

    def clear_pending_dual_responses(self, task_id: str) -> bool:
        """Clear pending dual responses for a task"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM pending_dual_responses WHERE task_id = ?
                """, (task_id,))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing pending dual responses: {e}")
            return False

    def update_dual_response_progress(self, task_id: str, source: str, content: str, 
                                    finished: bool = False) -> bool:
        """Update dual response progress (for streaming)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if source == 'local':
                    conn.execute("""
                        UPDATE pending_dual_responses 
                        SET local_response = ?, local_finished = ?
                        WHERE task_id = ?
                    """, (content, finished, task_id))
                elif source == 'gpt':
                    conn.execute("""
                        UPDATE pending_dual_responses 
                        SET gpt_response = ?, gpt_finished = ?
                        WHERE task_id = ?
                    """, (content, finished, task_id))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error updating dual response progress: {e}")
            return False 