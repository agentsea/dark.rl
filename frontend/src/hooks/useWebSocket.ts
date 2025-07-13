import { useState, useEffect, useRef, useCallback } from 'react'
import { flushSync } from 'react-dom'
import React from 'react'

export interface Message {
    role: 'user' | 'assistant' | 'system'
    content: string
    timestamp?: number
}

export interface OpenAIRequest {
    model?: string
    messages: Message[]
    stream: boolean
    max_tokens?: number
    temperature?: number
}

export interface StreamResponse {
    choices: Array<{
        delta: {
            content?: string
        }
        finish_reason?: string | null
    }>
}

export interface MCPServer {
    id: string
    name: string
    description: string
    requires_api_key: boolean
    api_key_available: boolean
    api_key_env?: string
}

export interface MCPServerRequest {
    type: 'list_mcp_servers'
    query?: string
}

export interface MCPServerActionsRequest {
    type: 'get_mcp_server_actions'
    server_ids: string[]
}

export interface MCPServerResponse {
    type: 'mcp_servers_response'
    servers: MCPServer[]
}

export interface MCPServerActionsResponse {
    type: 'mcp_server_actions_response'
    server_id: string
    actions: MCPServerAction[]
}

export interface MCPServerAction {
    name: string
    description: string
    parameters?: Record<string, any>
}

export enum ConnectionStatus {
    DISCONNECTED = 'disconnected',
    CONNECTING = 'connecting',
    CONNECTED = 'connected',
    ERROR = 'error'
}

interface UseWebSocketOptions {
    url: string
    autoConnect?: boolean
    reconnectAttempts?: number
    reconnectInterval?: number
}

export default function useWebSocket({
    url,
    autoConnect = true,
    reconnectAttempts = 3,
    reconnectInterval = 3000
}: UseWebSocketOptions) {
    const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>(
        ConnectionStatus.DISCONNECTED
    )
    const [messages, setMessages] = useState<Message[]>([])
    const [currentResponse, setCurrentResponse] = useState('')
    const [isStreaming, setIsStreaming] = useState(false)
    const [mcpServers, setMcpServers] = useState<MCPServer[]>([])
    const [loadingMcpServers, setLoadingMcpServers] = useState(false)
    const [mcpServerActions, setMcpServerActions] = useState<Record<string, MCPServerAction[]>>({})
    const [loadingMcpServerActions, setLoadingMcpServerActions] = useState(false)

    // Dual response state
    const [isDualResponse, setIsDualResponse] = useState(false);
    const [isDualTransitioning, setIsDualTransitioning] = useState(false);
    const [localResponse, setLocalResponse] = useState('');
    const [gptResponse, setGptResponse] = useState('');
    const [localFinished, setLocalFinished] = useState(false);
    const [gptFinished, setGptFinished] = useState(false);
    const [dualSessionId, setDualSessionId] = useState(0);
    const [localModel, setLocalModel] = useState('');
    const [gptModel, setGptModel] = useState('');

    // Refs for immediate updates
    const localResponseRef = useRef('');
    const gptResponseRef = useRef('');
    const currentSessionIdRef = useRef(0); // Add ref for immediate session ID updates
    const isTransitioningRef = useRef(false); // Add ref for immediate transition state

    const wsRef = useRef<WebSocket | null>(null)
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    const reconnectCountRef = useRef(0)
    const currentResponseRef = useRef('')

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return
        }

        console.log('Attempting to connect to:', url)
        setConnectionStatus(ConnectionStatus.CONNECTING)

        try {
            wsRef.current = new WebSocket(url)

            wsRef.current.onopen = () => {
                console.log('WebSocket connected successfully')
                setConnectionStatus(ConnectionStatus.CONNECTED)
                reconnectCountRef.current = 0
            }

            wsRef.current.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)

                    // Handle MCP server response
                    if (data.type === 'mcp_servers_response') {
                        const mcpResponse = data as MCPServerResponse
                        setMcpServers(mcpResponse.servers)
                        setLoadingMcpServers(false)
                        return
                    }

                    // Handle MCP server actions response
                    if (data.type === 'mcp_server_actions_response') {
                        const actionsResponse = data as MCPServerActionsResponse
                        setMcpServerActions(prev => ({
                            ...prev,
                            [actionsResponse.server_id]: actionsResponse.actions
                        }))
                        setLoadingMcpServerActions(false)
                        return
                    }

                    // Handle dual response start
                    if (data.type === 'dual_response_start') {
                        // CRITICAL: IMMEDIATELY hide any existing content before doing ANYTHING else
                        isTransitioningRef.current = true; // Set ref synchronously
                        setIsDualTransitioning(true); // Set state asynchronously
                        console.log('🔥 DUAL RESPONSE START - CONTENT IMMEDIATELY HIDDEN')

                        console.log('🔥 DUAL RESPONSE START DEBUG:')
                        console.log('🔥 Previous state before clearing:')
                        console.log('🔥   - dualSessionId:', dualSessionId)
                        console.log('🔥   - isDualResponse:', isDualResponse)
                        console.log('🔥   - localResponse length:', localResponse.length)
                        console.log('🔥   - localResponse content:', localResponse.substring(0, 100) + '...')
                        console.log('🔥   - gptResponse length:', gptResponse.length)
                        console.log('🔥   - gptResponse content:', gptResponse.substring(0, 100) + '...')
                        console.log('🔥   - localFinished:', localFinished)
                        console.log('🔥   - gptFinished:', gptFinished)
                        console.log('🔥   - localResponseRef.current length:', localResponseRef.current.length)
                        console.log('🔥   - gptResponseRef.current length:', gptResponseRef.current.length)

                        // Extract session ID from backend message
                        const newSessionId = data.session_id || Date.now();
                        console.log('🔥   - newSessionId from backend:', newSessionId)

                        // CRITICAL: Update session ref IMMEDIATELY before any other updates
                        currentSessionIdRef.current = newSessionId;
                        console.log('🔥   - currentSessionIdRef.current set to:', currentSessionIdRef.current)

                        // Clear refs immediately
                        localResponseRef.current = '';
                        gptResponseRef.current = '';
                        console.log('🔥   - Refs cleared')

                        // Then clear state with flushSync to force immediate update
                        flushSync(() => {
                            setIsDualResponse(true);
                            setLocalResponse('');
                            setGptResponse('');
                            setLocalFinished(false);
                            setGptFinished(false);
                            setLocalModel(data.local_model || 'local');
                            setGptModel(data.gpt_model || 'gpt-4.1');
                            setIsStreaming(true);

                            // Use session ID from backend and ensure state is updated
                            const newSessionId = data.session_id || Date.now();
                            console.log('🔥   - Setting dualSessionId: ', dualSessionId, ' -> ', newSessionId);
                            setDualSessionId(newSessionId);
                        });

                        // Log state after flushSync to verify it was updated
                        console.log('🔥 State after flushSync:')
                        console.log('🔥   - isDualResponse should be true')
                        console.log('🔥   - localResponse should be empty')
                        console.log('🔥   - gptResponse should be empty')
                        console.log('🔥   - dualSessionId should be updated to:', data.session_id)
                        console.log('🔥   - currentSessionIdRef.current:', currentSessionIdRef.current)

                        // End transition after a very short delay to ensure state is fully updated
                        // This prevents old content from flashing before new chunks arrive
                        setTimeout(() => {
                            console.log('🔥   - Ending dual transition after timeout')
                            isTransitioningRef.current = false; // Clear ref
                            setIsDualTransitioning(false); // Clear state
                        }, 50);

                        return;
                    }

                    // Handle all dual response chunks first - BEFORE session validation for debugging
                    if (data.type === 'dual_response_chunk') {
                        console.log('🔥 RAW CHUNK RECEIVED:')
                        console.log('🔥   - source:', data.source)
                        console.log('🔥   - content:', data.choices?.[0]?.delta?.content)
                        console.log('🔥   - finish_reason:', data.choices?.[0]?.finish_reason)
                        console.log('🔥   - chunkSessionId:', data.session_id)
                        console.log('🔥   - dualSessionId (state):', dualSessionId)
                        console.log('🔥   - currentSessionIdRef.current:', currentSessionIdRef.current)

                        // CRITICAL: Ignore chunks from old sessions to prevent flash of old content
                        const chunkSessionId = data.session_id || 0;
                        if (chunkSessionId !== currentSessionIdRef.current) {
                            console.log('🔥 IGNORING OLD CHUNK: session', chunkSessionId, 'vs current', currentSessionIdRef.current);
                            return;
                        }

                        const content = data.choices[0].delta?.content
                        console.log('🔥 PROCESSING CHUNK:')
                        console.log('🔥   - source:', data.source)
                        console.log('🔥   - content:', content)
                        console.log('🔥   - finish_reason:', data.choices[0].finish_reason)

                        if (data.source === 'local') {
                            if (content) {
                                // End transition on first content chunk to show new content immediately
                                if (isDualTransitioning) {
                                    console.log('🔥   - Ending transition on first LOCAL chunk')
                                    isTransitioningRef.current = false; // Clear ref
                                    setIsDualTransitioning(false); // Clear state
                                }

                                localResponseRef.current += content;
                                console.log('🔥   - localResponseRef updated: ', localResponse.length, ' -> ', localResponseRef.current.length);
                                console.log('🔥   - localResponseRef new content:', localResponseRef.current.substring(0, 200) + '...');
                                setLocalResponse(localResponseRef.current);
                            }
                            if (data.choices[0].finish_reason) {
                                console.log('🔥 LOCAL RESPONSE FINISHED with reason:', data.choices[0].finish_reason);
                                console.log('🔥   - Setting localFinished to TRUE');
                                setLocalFinished(true);
                            }
                        } else if (data.source === 'gpt') {
                            if (content) {
                                // End transition on first content chunk to show new content immediately
                                if (isDualTransitioning) {
                                    console.log('🔥   - Ending transition on first GPT chunk')
                                    isTransitioningRef.current = false; // Clear ref
                                    setIsDualTransitioning(false); // Clear state
                                }

                                gptResponseRef.current += content;
                                console.log('🔥   - gptResponseRef updated: ', gptResponse.length, ' -> ', gptResponseRef.current.length);
                                console.log('🔥   - gptResponseRef new content:', gptResponseRef.current.substring(0, 200) + '...');
                                setGptResponse(gptResponseRef.current);
                            }
                            if (data.choices[0].finish_reason) {
                                console.log('🔥 GPT RESPONSE FINISHED with reason:', data.choices[0].finish_reason);
                                console.log('🔥   - Setting gptFinished to TRUE');
                                setGptFinished(true);
                            }
                        }
                        return;
                    }

                    // Handle dual response complete
                    if (data.type === 'dual_response_complete') {
                        // CRITICAL: Ignore completion messages from old sessions
                        const completionSessionId = data.session_id || 0;
                        if (completionSessionId !== currentSessionIdRef.current) {
                            console.log('🔥 IGNORING OLD COMPLETION: session', completionSessionId, 'vs current', currentSessionIdRef.current);
                            return;
                        }

                        console.log('🔥 DUAL RESPONSE COMPLETE DEBUG:')
                        console.log('🔥   - localResponse final length:', localResponse.length)
                        console.log('🔥   - gptResponse final length:', gptResponse.length)
                        console.log('🔥   - localFinished before:', localFinished)
                        console.log('🔥   - gptFinished before:', gptFinished)
                        console.log('🔥   - backend reports local_finished:', data.local_finished)
                        console.log('🔥   - backend reports gpt_finished:', data.gpt_finished)
                        console.log('🔥   - session_id:', data.session_id)

                        // Force set finished states based on backend completion
                        if (data.local_finished !== undefined) {
                            console.log('🔥   - Force setting localFinished to:', data.local_finished);
                            setLocalFinished(data.local_finished);
                        }
                        if (data.gpt_finished !== undefined) {
                            console.log('🔥   - Force setting gptFinished to:', data.gpt_finished);
                            setGptFinished(data.gpt_finished);
                        }

                        setIsStreaming(false);
                        console.log('🔥   - Set isStreaming to false')
                        console.log('🔥   - Dual response completion processing done')
                        return;
                    }

                    // Handle streaming response (single model)
                    const streamData = data as StreamResponse
                    if (streamData.choices && streamData.choices[0]) {
                        const choice = streamData.choices[0]

                        if (choice.delta?.content) {
                            currentResponseRef.current += choice.delta.content
                            setCurrentResponse(currentResponseRef.current)
                            setIsStreaming(true)
                        }

                        if (choice.finish_reason === 'stop' || choice.finish_reason === 'length') {
                            console.log('Stream finished, finalizing message')

                            // Add any final content chunk
                            if (choice.delta?.content) {
                                currentResponseRef.current += choice.delta.content
                                setCurrentResponse(currentResponseRef.current)
                            }

                            // Check if this message contains a tool call
                            const toolCallMatch = currentResponseRef.current.match(/<tool_call>(.*?)<\/tool_call>/s)
                            const thinkingMatch = currentResponseRef.current.match(/<think>(.*?)<\/think>/s)

                            if (toolCallMatch) {
                                console.log('Tool call detected in completed message, showing modal')

                                try {
                                    const toolCall = JSON.parse(toolCallMatch[1])
                                    const thinking = thinkingMatch ? thinkingMatch[1].trim() : undefined

                                    // Trigger tool call modal
                                    const toolCallEvent = new CustomEvent('toolCallRequested', {
                                        detail: {
                                            toolCall,
                                            thinking,
                                            fullResponse: currentResponseRef.current
                                        }
                                    })
                                    window.dispatchEvent(toolCallEvent)

                                    // End streaming but don't clear response yet (tool modal will handle continuation)
                                    setIsStreaming(false)
                                    return
                                } catch (e) {
                                    console.error('Error parsing tool call:', e)
                                }
                            }

                            // No tool call detected, handle as normal message
                            // End streaming
                            setIsStreaming(false)

                            // Add the final message to history
                            setMessages(prev => [
                                ...prev,
                                {
                                    role: 'assistant',
                                    content: currentResponseRef.current,
                                    timestamp: Date.now()
                                }
                            ])

                            // Clear the response and reset for next message
                            currentResponseRef.current = ''
                            // Keep currentResponse visible until user continues conversation
                        }
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error, 'Raw data:', event.data)
                    setConnectionStatus(ConnectionStatus.ERROR)
                }
            }

            wsRef.current.onclose = (event) => {
                setConnectionStatus(ConnectionStatus.DISCONNECTED)
                setIsStreaming(false)
                console.log('WebSocket disconnected:', event.code, event.reason)

                // Attempt to reconnect
                if (reconnectCountRef.current < reconnectAttempts) {
                    reconnectCountRef.current++
                    reconnectTimeoutRef.current = setTimeout(() => {
                        console.log(`Reconnecting... attempt ${reconnectCountRef.current}`)
                        connect()
                    }, reconnectInterval)
                }
            }

            wsRef.current.onerror = (error) => {
                setConnectionStatus(ConnectionStatus.ERROR)
                console.error('WebSocket error:', error)
            }

        } catch (error) {
            setConnectionStatus(ConnectionStatus.ERROR)
            console.error('Failed to create WebSocket connection:', error)
        }
    }, [url, reconnectAttempts, reconnectInterval])

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current)
            reconnectTimeoutRef.current = null
        }

        if (wsRef.current) {
            wsRef.current.close()
            wsRef.current = null
        }

        setConnectionStatus(ConnectionStatus.DISCONNECTED)
        reconnectCountRef.current = 0
    }, [])

    const sendMessage = useCallback((content: string, model = 'default', customMessages?: Message[], taskId?: string, isAutoResponse = false) => {
        console.log(`🐛 [useWebSocket] sendMessage called with content="${content}", model=${model}, taskId=${taskId}, isAutoResponse=${isAutoResponse}`)
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('🐛 [useWebSocket] WebSocket not connected for sending message')
            return
        }

        const userMessage: Message = {
            role: 'user',
            content,
            timestamp: Date.now()
        }

        // Use custom messages if provided, otherwise use stored messages
        const messagesToSend = customMessages || [...messages, userMessage]

        // Add user message to chat only if not using custom messages
        if (!customMessages) {
            setMessages(prev => [...prev, userMessage])
        }

        // Prepare OpenAI-format request
        const request: OpenAIRequest & { task_id?: string; auto_response?: boolean } = {
            model,
            messages: messagesToSend,
            stream: true,
            max_tokens: 2048,
            temperature: 0.7,
            ...(taskId && { task_id: taskId }),
            ...(isAutoResponse && { auto_response: true })
        }

        console.log(`🐛 [useWebSocket] Sending message request:`, request)

        // Send to WebSocket
        wsRef.current.send(JSON.stringify(request))
        currentResponseRef.current = ''
        setCurrentResponse('')
        setIsStreaming(true)
    }, [connectionStatus, messages])

    const clearMessages = useCallback(() => {
        setMessages([])
        currentResponseRef.current = ''
        setCurrentResponse('')
        setIsStreaming(false)
    }, [])

    const clearCurrentResponse = useCallback(() => {
        currentResponseRef.current = ''
        setCurrentResponse('')
    }, [])

    const clearDualResponse = useCallback(() => {
        console.log('🔥 CLEAR DUAL RESPONSE DEBUG:')
        console.log('🔥 State before clearing:')
        console.log('🔥   - isDualResponse:', isDualResponse)
        console.log('🔥   - localResponse length:', localResponse.length)
        console.log('🔥   - gptResponse length:', gptResponse.length)
        console.log('🔥   - localModel:', localModel)
        console.log('🔥   - gptModel:', gptModel)
        console.log('🔥   - localFinished:', localFinished)
        console.log('🔥   - gptFinished:', gptFinished)
        console.log('🔥   - localResponseRef.current length:', localResponseRef.current.length)
        console.log('🔥   - gptResponseRef.current length:', gptResponseRef.current.length)

        // Start transition
        setIsDualTransitioning(true);

        // Clear refs immediately
        localResponseRef.current = '';
        gptResponseRef.current = '';
        currentSessionIdRef.current = 0; // Reset session ref too

        // End transition state
        setIsDualTransitioning(false);

        // Clear state
        setIsDualResponse(false);
        setLocalResponse('');
        setGptResponse('');
        setLocalFinished(false);
        setGptFinished(false);
        setDualSessionId(0);
        setLocalModel('');
        setGptModel('');
        setIsStreaming(false);

        console.log('🔥 State after clearing:')
        console.log('🔥   - isDualResponse:', false)
        console.log('🔥   - localResponseRef.current length:', localResponseRef.current.length)
        console.log('🔥   - gptResponseRef.current length:', gptResponseRef.current.length)
    }, [isDualResponse, localResponse.length, gptResponse.length, localModel, gptModel, localFinished, gptFinished, localResponseRef, gptResponseRef]);

    const selectModel = useCallback(async (selectedModel: 'local' | 'gpt', taskId?: string) => {
        console.log(`🐛 [useWebSocket] selectModel called with model=${selectedModel}, taskId=${taskId}`)

        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('🐛 [useWebSocket] WebSocket not connected for model selection')
            return
        }

        // 🔥 CRITICAL FIX: Capture current responses BEFORE clearing
        const currentLocalResponse = localResponseRef.current
        const currentGptResponse = gptResponseRef.current

        // 🔥 CRITICAL FIX: Clear content IMMEDIATELY when user selects to prevent flash
        console.log('🔥 IMMEDIATE CLEAR: Clearing all content before sending request')
        setLocalResponse('')
        setGptResponse('')
        localResponseRef.current = ''
        gptResponseRef.current = ''
        setLocalFinished(false)
        setGptFinished(false)
        setIsDualResponse(false)
        setIsDualTransitioning(true)
        isTransitioningRef.current = true

        const request = {
            type: 'model_selection',
            selected_model: selectedModel,
            local_response: currentLocalResponse,
            gpt_response: currentGptResponse,
            local_model_name: localModel,
            gpt_model_name: gptModel,
            task_id: taskId
        }

        console.log(`🐛 [useWebSocket] Sending model selection request:`, request)

        return new Promise((resolve, reject) => {
            const originalOnMessage = wsRef.current?.onmessage

            const handleResponse = (event: MessageEvent) => {
                try {
                    const data = JSON.parse(event.data)
                    console.log(`🐛 [useWebSocket] Received response in selectModel:`, data.type)

                    if (data.type === 'model_selection_response') {
                        console.log(`🐛 [useWebSocket] Model selection response received, success=${data.success}`)
                        // Restore original handler
                        if (wsRef.current && originalOnMessage) {
                            wsRef.current.onmessage = originalOnMessage
                        }
                        resolve(data)
                        return
                    }
                } catch (error) {
                    console.error('🐛 [useWebSocket] Error parsing selectModel response:', error)
                }

                // Pass other messages to original handler
                if (originalOnMessage) {
                    originalOnMessage(event)
                }
            }

            // Set temporary handler
            if (wsRef.current) {
                wsRef.current.onmessage = handleResponse
            }

            // Send the request
            wsRef.current.send(JSON.stringify(request))
            console.log(`🐛 [useWebSocket] Model selection request sent`)

            // Set timeout for response
            setTimeout(() => {
                if (wsRef.current && originalOnMessage) {
                    wsRef.current.onmessage = originalOnMessage
                }
                reject(new Error('Model selection timeout'))
            }, 5000)
        })
    }, [localModel, gptModel])

    const getMcpServers = useCallback((query = '') => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for MCP server request')
            return
        }

        setLoadingMcpServers(true)

        const request: MCPServerRequest = {
            type: 'list_mcp_servers',
            query
        }

        wsRef.current.send(JSON.stringify(request))
    }, [connectionStatus])

    const getMcpServerActions = useCallback((serverIds: string[]) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for MCP server actions request')
            return
        }

        setLoadingMcpServerActions(true)

        const request: MCPServerActionsRequest = {
            type: 'get_mcp_server_actions',
            server_ids: serverIds
        }

        wsRef.current.send(JSON.stringify(request))
    }, [connectionStatus])

    const createTask = useCallback(async (prompt: string, model = 'qwen2.5-vl'): Promise<string | null> => {
        // Wait for connection if it's still connecting
        let retries = 0
        const maxRetries = 30 // 3 seconds with 100ms intervals

        while (retries < maxRetries) {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                break
            }
            if (wsRef.current?.readyState === WebSocket.CLOSED ||
                wsRef.current?.readyState === WebSocket.CLOSING) {
                console.error('WebSocket connection failed for task creation')
                return null
            }
            await new Promise(resolve => setTimeout(resolve, 100))
            retries++
        }

        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for task creation after waiting')
            return null
        }

        return new Promise((resolve) => {
            const request = {
                type: 'create_task',
                prompt,
                model
            }

            // Set up one-time listener for task creation response
            const ws = wsRef.current!  // We know it's not null from the check above
            const originalOnMessage = ws.onmessage

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)

                    if (data.type === 'task_created') {
                        // Restore original message handler
                        if (wsRef.current) {
                            wsRef.current.onmessage = originalOnMessage
                        }
                        resolve(data.task_id)
                        return
                    }

                    if (data.type === 'error') {
                        console.error('Task creation error:', data.error)
                        if (wsRef.current) {
                            wsRef.current.onmessage = originalOnMessage
                        }
                        resolve(null)
                        return
                    }

                    // Pass other messages to original handler
                    if (originalOnMessage) {
                        originalOnMessage.call(ws, event)
                    }
                } catch (error) {
                    console.error('Error parsing task creation response:', error)
                    if (wsRef.current) {
                        wsRef.current.onmessage = originalOnMessage
                    }
                    resolve(null)
                }
            }

            ws.send(JSON.stringify(request))
        })
    }, [connectionStatus])

    const getTask = useCallback(async (taskId: string): Promise<any> => {
        console.log(`🐛 [useWebSocket] getTask called with taskId=${taskId}`)

        // Wait for connection if it's still connecting
        let retries = 0
        const maxRetries = 30 // 3 seconds with 100ms intervals

        while (retries < maxRetries) {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                break
            }
            if (wsRef.current?.readyState === WebSocket.CLOSED ||
                wsRef.current?.readyState === WebSocket.CLOSING) {
                console.error('🐛 [useWebSocket] WebSocket connection failed for task retrieval')
                return null
            }
            await new Promise(resolve => setTimeout(resolve, 100))
            retries++
        }

        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('🐛 [useWebSocket] WebSocket not connected for task retrieval after waiting')
            return null
        }

        return new Promise((resolve) => {
            const request = {
                type: 'get_task',
                task_id: taskId
            }

            console.log(`🐛 [useWebSocket] Sending getTask request:`, request)

            // Set up one-time listener for task data response
            const ws = wsRef.current!
            const originalOnMessage = ws.onmessage

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)
                    console.log(`🐛 [useWebSocket] Received response in getTask:`, data.type)

                    if (data.type === 'task_data') {
                        console.log(`🐛 [useWebSocket] Task data received, messages count=${data.messages?.length || 0}`)
                        // Restore original message handler
                        if (wsRef.current) {
                            wsRef.current.onmessage = originalOnMessage
                        }
                        resolve(data)
                        return
                    }

                    if (data.type === 'error') {
                        console.error('🐛 [useWebSocket] Task retrieval error:', data.error)
                        if (wsRef.current) {
                            wsRef.current.onmessage = originalOnMessage
                        }
                        resolve({ error: data.error })
                        return
                    }

                    // Pass other messages to original handler
                    if (originalOnMessage) {
                        originalOnMessage.call(ws, event)
                    }
                } catch (error) {
                    console.error('🐛 [useWebSocket] Error parsing task retrieval response:', error)
                    if (wsRef.current) {
                        wsRef.current.onmessage = originalOnMessage
                    }
                    resolve({ error: 'Failed to parse task data' })
                }
            }

            ws.send(JSON.stringify(request))
        })
    }, [connectionStatus])

    // WebSocket connection and message handling
    useEffect(() => {
        if (autoConnect) {
            connect()
        }

        return () => {
            disconnect()
        }
    }, [autoConnect, connect, disconnect])

    const sendLearningFeedback = useCallback(async (type: string, message: any, taskId: string, userComment?: string, messageIndex?: number): Promise<void> => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for learning feedback')
            return
        }

        const request = {
            type: 'learning_feedback',
            feedback_type: type,
            message: message,
            task_id: taskId,
            message_index: messageIndex,
            user_comment: userComment
        }

        wsRef.current.send(JSON.stringify(request))
    }, [connectionStatus])

    const executeMcpAction = useCallback(async (serverId: string, actionName: string, parameters: Record<string, any>): Promise<any> => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for MCP action execution')
            return null
        }

        return new Promise((resolve) => {
            const request = {
                type: 'execute_mcp_action',
                server_id: serverId,
                action_name: actionName,
                parameters: parameters
            }

            // Set up one-time listener for action result
            const ws = wsRef.current!
            const originalOnMessage = ws.onmessage

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)

                    if (data.type === 'mcp_action_result') {
                        // Restore original message handler
                        if (wsRef.current) {
                            wsRef.current.onmessage = originalOnMessage
                        }
                        resolve(data)
                        return
                    }

                    // Pass other messages to original handler
                    if (originalOnMessage) {
                        originalOnMessage.call(ws, event)
                    }
                } catch (error) {
                    console.error('Error parsing MCP action response:', error)
                    if (wsRef.current) {
                        wsRef.current.onmessage = originalOnMessage
                    }
                    resolve({ error: 'Failed to parse response' })
                }
            }

            ws.send(JSON.stringify(request))
        })
    }, [connectionStatus])

    const executeToolAndContinue = useCallback(async (
        serverId: string,
        toolName: string,
        parameters: Record<string, any>,
        currentResponse: string,
        mentionedServers: string[],
        model: string = 'qwen2.5-vl',
        taskId?: string
    ): Promise<void> => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for tool execution and continuation')
            return
        }

        const request = {
            type: 'execute_tool_and_continue',
            server_id: serverId,
            tool_name: toolName,
            parameters: parameters,
            current_response: currentResponse,
            mentioned_servers: mentionedServers,
            model: model,
            task_id: taskId
        }

        // Resume streaming mode
        setIsStreaming(true)

        wsRef.current.send(JSON.stringify(request))
    }, [connectionStatus])

    const sendCorrectionWithExecution = useCallback(async (
        taskId: string,
        messageIndex: number,
        correctedToolCall: { name: string; arguments: Record<string, any> },
        thought: string,
        shouldExecute: boolean = false
    ): Promise<any> => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for correction submission')
            return null
        }

        return new Promise((resolve) => {
            const request = {
                type: 'correction_with_execution',
                task_id: taskId,
                message_index: messageIndex,
                corrected_tool_call: correctedToolCall,
                thought: thought,
                should_execute: shouldExecute
            }

            // Set up one-time listener for correction response
            const ws = wsRef.current!
            const originalOnMessage = ws.onmessage

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)

                    if (data.type === 'correction_processed') {
                        // Restore original message handler
                        if (wsRef.current) {
                            wsRef.current.onmessage = originalOnMessage
                        }
                        resolve(data)
                        return
                    }

                    if (data.type === 'error') {
                        console.error('Correction error:', data.error)
                        if (wsRef.current) {
                            wsRef.current.onmessage = originalOnMessage
                        }
                        resolve({ error: data.error })
                        return
                    }

                    // Pass other messages to original handler
                    if (originalOnMessage) {
                        originalOnMessage.call(ws, event)
                    }
                } catch (error) {
                    console.error('Error parsing correction response:', error)
                    if (wsRef.current) {
                        wsRef.current.onmessage = originalOnMessage
                    }
                    resolve({ error: 'Failed to parse response' })
                }
            }

            ws.send(JSON.stringify(request))
        })
    }, [connectionStatus])

    return {
        connectionStatus,
        messages,
        currentResponse,
        isStreaming,
        mcpServers,
        loadingMcpServers,
        mcpServerActions,
        loadingMcpServerActions,
        // Dual response state
        isDualResponse: isDualResponse && !isDualTransitioning, // Hide during transition
        isDualTransitioning,
        isTransitioningRef, // Add ref for immediate transition checking
        localResponse,
        gptResponse,
        localModel,
        gptModel,
        localFinished,
        gptFinished,
        dualSessionId,
        connect,
        disconnect,
        sendMessage,
        clearMessages,
        getMcpServers,
        getMcpServerActions,
        createTask,
        getTask,
        clearCurrentResponse,
        clearDualResponse,
        selectModel,
        sendLearningFeedback,
        executeMcpAction,
        executeToolAndContinue,
        sendCorrectionWithExecution
    }
} 