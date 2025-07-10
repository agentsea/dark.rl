import { useState, useEffect, useRef, useCallback } from 'react'

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

                    // Handle streaming response
                    const streamData = data as StreamResponse
                    if (streamData.choices && streamData.choices[0]) {
                        const choice = streamData.choices[0]

                        if (choice.delta?.content) {
                            currentResponseRef.current += choice.delta.content
                            setCurrentResponse(currentResponseRef.current)
                            setIsStreaming(true)
                        }

                        if (choice.finish_reason === 'tool_call_requested') {
                            console.log('Tool call requested, showing modal')

                            // Add any final content chunk
                            if (choice.delta?.content) {
                                currentResponseRef.current += choice.delta.content
                                setCurrentResponse(currentResponseRef.current)
                            }

                            // Extract tool call and thinking
                            const toolCallMatch = currentResponseRef.current.match(/<tool_call>(.*?)<\/tool_call>/s)
                            const thinkingMatch = currentResponseRef.current.match(/<think>(.*?)<\/think>/s)

                            if (toolCallMatch) {
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

                                    // Pause streaming - don't finalize message yet
                                    setIsStreaming(false)
                                    return
                                } catch (e) {
                                    console.error('Error parsing tool call:', e)
                                }
                            }
                        }

                        if (choice.finish_reason === 'stop' || choice.finish_reason === 'length') {
                            console.log('Stream finished, finalizing message')

                            // Add any final content chunk
                            if (choice.delta?.content) {
                                currentResponseRef.current += choice.delta.content
                                setCurrentResponse(currentResponseRef.current)
                            }

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
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for sending message')
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
        // Wait for connection if it's still connecting
        let retries = 0
        const maxRetries = 30 // 3 seconds with 100ms intervals

        while (retries < maxRetries) {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                break
            }
            if (wsRef.current?.readyState === WebSocket.CLOSED ||
                wsRef.current?.readyState === WebSocket.CLOSING) {
                console.error('WebSocket connection failed for task retrieval')
                return null
            }
            await new Promise(resolve => setTimeout(resolve, 100))
            retries++
        }

        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for task retrieval after waiting')
            return null
        }

        return new Promise((resolve) => {
            const request = {
                type: 'get_task',
                task_id: taskId
            }

            // Set up one-time listener for task data response
            const ws = wsRef.current!
            const originalOnMessage = ws.onmessage

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)

                    if (data.type === 'task_data') {
                        // Restore original message handler
                        if (wsRef.current) {
                            wsRef.current.onmessage = originalOnMessage
                        }
                        resolve(data)
                        return
                    }

                    if (data.type === 'error') {
                        console.error('Task retrieval error:', data.error)
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
                    console.error('Error parsing task retrieval response:', error)
                    if (wsRef.current) {
                        wsRef.current.onmessage = originalOnMessage
                    }
                    resolve({ error: 'Failed to parse task data' })
                }
            }

            ws.send(JSON.stringify(request))
        })
    }, [connectionStatus])

    useEffect(() => {
        if (autoConnect) {
            connect()
        }

        return () => {
            disconnect()
        }
    }, [autoConnect, connect, disconnect])

    const sendLearningFeedback = useCallback(async (type: string, message: any, taskId: string, userComment?: string): Promise<void> => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for learning feedback')
            return
        }

        const request = {
            type: 'learning_feedback',
            feedback_type: type,
            message: message,
            task_id: taskId,
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
        connect,
        disconnect,
        sendMessage,
        clearMessages,
        getMcpServers,
        getMcpServerActions,
        createTask,
        getTask,
        clearCurrentResponse,
        sendLearningFeedback,
        executeMcpAction,
        executeToolAndContinue,
        sendCorrectionWithExecution
    }
} 