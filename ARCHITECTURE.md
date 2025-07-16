# Dark.RL Architecture Specification

This document provides a detailed specification of the Dark.RL application, covering the frontend, backend, and the interaction patterns between them.

## 1. High-Level Overview

Dark.RL is a web-based platform for interacting with and training Large Language Models (LLMs), with a specific focus on **tool usage** and **agent-like behavior**. It provides a user interface for giving tasks to an AI, which can then use a predefined set of external tools (via the Model Context Protocol) to accomplish its goals.

Key features include:
- **Task-based conversations:** Interactions are organized into persistent tasks.
- **Dual Model Response:** The UI can present responses from two different models (a local model and a GPT model) side-by-side, allowing the user to choose the better one.
- **Tool Integration (MCP):** The AI can call external tools (e.g., for filesystem access, web browsing with Playwright).
- **Interactive Correction:** Users can correct a model's flawed tool usage, and this correction is used for online learning.
- **Autonomous Operation:** The system is designed to allow the AI to chain multiple tool calls together with minimal user intervention, creating an "agent loop".

---

## 2. System Components

The system consists of four main components:

1.  **Frontend:** A single-page application (SPA) built with **React**, **Vite**, **TypeScript**, and styled with **Tailwind CSS**. It is responsible for rendering the user interface and communicating with the backend.
2.  **Backend (WebSocket Server):** A Python server using `websockets` and `asyncio` (`websocket_server.py`). It serves as the central hub, managing WebSocket connections, orchestrating the LLM, handling tasks, and communicating with MCP servers.
3.  **Database:** An **SQLite** database (`dark_rl_tasks.db`) used for persisting all task-related data, including user prompts, AI responses, and conversation history.
4.  **MCP Servers:** External, standalone **Node.js** processes that expose tools to the backend via the Model Context Protocol (MCP). These are the "tools" the AI can use.

![System Diagram](https://i.imgur.com/example.png)  *Placeholder for a future diagram*

---

## 3. Frontend Architecture

The frontend code is located in the `/frontend` directory.

### 3.1. Core Technologies

-   **React:** For building the user interface.
-   **React Router:** For handling client-side routing (`/` for the landing page, `/tasks/:id` for the task view).
-   **TypeScript:** For type safety.
-   **`useWebSocket` (Custom Hook):** The most important piece of the frontend architecture. This hook, located in `frontend/src/hooks/useWebSocket.ts`, encapsulates all WebSocket communication logic and state management.
-   **Tailwind CSS:** For styling.

### 3.2. Component Structure

-   `App.tsx`: The root component that sets up the routing.
-   `LandingPage.tsx`: The home page, which likely shows a list of recent tasks and allows for the creation of new ones.
-   `TaskPage.tsx`: The main interaction view for a single task. It displays the conversation history and the dual response UI.
-   `CorrectionModal.tsx` & `ToolCallModal.tsx`: Modals for handling user interaction with the AI's tool calls.

### 3.3. State Management & Data Flow

State is managed almost exclusively within the `useWebSocket.ts` hook and passed down to the components. This hook is a "headless" state machine for the application.

-   **`connectionStatus`**: Tracks the WebSocket connection state (`CONNECTING`, `CONNECTED`, `DISCONNECTED`).
-   **`taskMessages`**: An array of `Message` objects representing the current conversation history for a task.
-   **`isDualResponse`, `localResponse`, `gptResponse`**: State variables that control the dual response UI. When `isDualResponse` is true, the `TaskPage` renders two columns to display the streaming responses from the local and GPT models.
-   **Functions:** The hook returns functions that components can call to interact with the backend (e.g., `sendMessage`, `createTask`, `getTask`, `selectModel`).

When a component calls a function like `sendMessage`, the hook constructs the appropriate JSON payload and sends it over the WebSocket. When a message is received from the backend, the `onmessage` handler within the hook parses it, updates the relevant state, and causes the UI to re-render.

### 3.4. Browser Interaction and Rendering

-   **Content Parsing:** The `TaskPage` component contains a powerful `parseContent` utility function. It scans the `content` of each message for special XML-like tags:
    -   `<think>...</think>`: Renders the text within as the AI's reasoning, often styled differently (e.g., italicized, indented).
    -   `<tool_call>...</tool_call>`: Parses the JSON content and renders it as a formatted code block, clearly indicating a tool is being called.
    -   `<tool_response>...</tool_response>`: Renders the result from a tool execution, often with a success or failure indicator.
-   **Dual Response UI:** This is the most complex UI feature. The `TaskPage` uses the `isDualResponse`, `localResponse`, `gptResponse`, `localFinished`, and `gptFinished` state from the `useWebSocket` hook to render two side-by-side panels. As chunks of data arrive from the WebSocket, the `localResponse` and `gptResponse` states are updated, causing the text to appear to "stream" into the UI. Once both are finished, the panels become clickable, allowing the user to make a selection.

---

## 4. Backend Architecture (`websocket_server.py`)

The backend is a single Python file that runs an asynchronous WebSocket server.

### 4.1. Core Class: `DarkRLLLMServer`

This class is a singleton that holds the server's state and logic. Its key responsibilities include:
-   Initializing and holding the `AsyncOnlineLLM` instance.
-   Managing the `TaskManager` for database interactions.
-   Handling MCP server configurations and maintaining a cache of `MCPClient` connections.
-   Containing all the logic for handling different types of requests from the frontend.

### 4.2. WebSocket Handling (`handle_client`)

This async function is the entry point for every new client connection. It contains an `async for message in websocket:` loop that listens for incoming messages.

Inside the loop, it performs the following steps:
1.  Parses the incoming message as JSON.
2.  Reads the `type` field of the JSON object.
3.  Uses a series of `if/elif` statements to route the request to the appropriate handler method within the `DarkRLLLMServer` class.

### 4.3. Key Backend Flows

-   **Streaming Responses:** The `stream_response` method is the orchestrator for generating AI responses.
    -   It first determines if it should be in "action mode" by checking for `@server` mentions in the user's prompt (`should_use_action_mode`).
    -   If an OpenAI API key is available, it defaults to `stream_dual_response` or `stream_dual_action_response`. These methods start two concurrent asyncio tasks: one for the local LLM and one for GPT.
    -   As each model generates tokens, the backend sends `dual_response_chunk` messages to the frontend.
    -   When both models are done, it sends a `dual_response_complete` message.
-   **Task Management:** The `TaskManager` class provides a simple ORM-like interface to the SQLite database. The server uses it to `create_task`, `get_task`, `add_message`, etc.
-   **MCP Action Execution:** The `execute_mcp_action` method is responsible for:
    1.  Finding the correct `MCPClient` for the requested `server_id`.
    2.  Calling the `session.connector.call_tool` method with the action name and parameters.
    3.  Serializing the result from the MCP server into a JSON-friendly format.
    4.  Returning the result to the caller.

---

## 5. Data Models & Payloads

Communication is done via JSON objects with a `type` field indicating the message's purpose.

### 5.1. Frontend -> Backend

| Type                          | Description                                                                                              | Payload Example                                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `create_task`                 | Creates a new task.                                                                                      | `{ "type": "create_task", "prompt": "Find the best restaurants in Boulder." }`                                     |
| `get_task`                    | Retrieves a task and its messages.                                                                       | `{ "type": "get_task", "task_id": "..." }`                                                                         |
| (Chat Request)                | The standard request to get a response from the AI. Sent by `sendMessage`.                               | `{ "messages": [...], "model": "qwen3", "stream": true, "task_id": "..." }`                                       |
| `model_selection`             | Sent after the user chooses a response from the dual view.                                               | `{ "type": "model_selection", "task_id": "...", "selected_model": "local", "local_response": "...", "gpt_response": "..." }` |
| `correction_with_execution`   | Sent from the Correction Modal. Provides a corrected tool call and tells the backend to execute it.      | `{ "type": "correction_with_execution", "task_id": "...", "corrected_tool_call": {...}, "thought": "..." }`       |
| `learning_feedback`           | Sent when a user approves a response or provides a comment. Used for online training.                    | `{ "type": "learning_feedback", "feedback_type": "approve", "message": {...}, "task_id": "..." }`                   |
| `list_mcp_servers`            | Requests the list of available MCP servers.                                                              | `{ "type": "list_mcp_servers" }`                                                                                   |
| `get_mcp_server_actions`      | Requests the available actions for specific MCP servers.                                                 | `{ "type": "get_mcp_server_actions", "server_ids": ["playwright", "filesystem"] }`                                 |

### 5.2. Backend -> Frontend

| Type                       | Description                                                                     | Payload Example                                                                            |
| -------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `task_created`             | Response to `create_task`, returning the new task's ID.                         | `{ "type": "task_created", "task_id": "..." }`                                             |
| `task_data`                | Response to `get_task`, containing the full task details and message history.   | `{ "type": "task_data", "task": {...}, "messages": [...] }`                                 |
| `dual_response_start`      | Signals the beginning of a dual response stream.                                | `{ "type": "dual_response_start", "session_id": 123, "local_model": "qwen3" }`               |
| `dual_response_chunk`      | A chunk of a streaming response for one of the dual models.                     | `{ "type": "dual_response_chunk", "source": "local", "choices": [{"delta": {"content": "..."}}] }` |
| `dual_response_complete`   | Signals the end of a dual response stream.                                      | `{ "type": "dual_response_complete", "session_id": 123 }`                                  |
| `model_selection_response` | Confirms that the user's model selection was received and processed.            | `{ "type": "model_selection_response", "success": true }`                                  |
| `tool_result`              | Pushes the result of a tool execution to the client as a new message.           | `{ "type": "tool_result", "task_id": "...", "message": { "role": "user", "content": "<tool_response>..." }}` |
| `mcp_servers_response`     | Returns the list of available MCP servers.                                      | `{ "type": "mcp_servers_response", "servers": [...] }`                                      |
| `mcp_action_result`        | The direct result of an `execute_mcp_action` request.                           | `{ "type": "mcp_action_result", "success": true, "result": {...} }`                         |

---

## 6. Core Interaction Flow: The Agent Loop

This sequence describes the primary user-to-AI interaction pattern.

1.  **Task Creation:** A user provides an initial prompt on the `LandingPage` or directly. A `create_task` request is sent to the backend, which creates a new entry in the database and returns a `task_id`. The user is navigated to `/tasks/:id`.

2.  **Initial AI Response (Auto-Trigger):** The `TaskPage` loads. Its `useEffect` hook determines that this is a new task with only one user message. It automatically calls `sendMessage` to get the first response from the AI.

3.  **Dual Response Generation:** The backend receives the request and starts the dual response stream (`stream_dual_action_response`). It sends a `dual_response_start` message, followed by a series of `dual_response_chunk` messages for both the local model and GPT. The AI is prompted to respond with `<think>` and `<tool_call>` tags.

4.  **UI Rendering:** The `TaskPage` renders the two streaming responses in the UI. The content is parsed to show the thinking and tool calls in a user-friendly way.

5.  **User Selection or Correction:**
    *   **Selection:** Once both streams finish, the user clicks the better response. `handleModelSelection` is called, which sends a `model_selection` message to the backend.
    *   **Correction:** If a response is flawed, the user can click the "Correct" icon to open the `CorrectionModal`. They can then build a valid tool call. Submitting the modal sends a `correction_with_execution` request to the backend.

6.  **Backend Processing & Tool Execution:**
    *   The backend receives the `model_selection` or `correction_with_execution` message.
    *   It saves the chosen/corrected assistant message to the database.
    *   It parses the `<tool_call>` from the message content.
    *   It calls `execute_mcp_action` to run the tool on the appropriate external MCP server.

7.  **Result Persistence & Continuation:**
    *   The result from the MCP action is received by the backend.
    *   The backend formats this result into a new message with `<tool_response>` tags.
    *   This new "user" message (representing the environment's feedback) is added to the task's message history in the database.
    *   The backend pushes this new message to the client via a `tool_result` message for a seamless UI update.
    *   The `TaskPage` receives the tool result and its `useEffect` hook detects that the last message is a tool response, automatically triggering another call to `sendMessage` to continue the loop.

This loop continues, with the AI making further tool calls based on previous results, until it decides the task is complete and responds without a `<tool_call>`. 