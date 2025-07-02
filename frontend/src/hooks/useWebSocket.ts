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
}

export interface MCPServerRequest {
    type: 'list_mcp_servers'
    query?: string
}

export interface MCPServerResponse {
    type: 'mcp_servers_response'
    servers: MCPServer[]
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

    const wsRef = useRef<WebSocket | null>(null)
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    const reconnectCountRef = useRef(0)

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
                    console.log('Received WebSocket message:', event.data)
                    const data = JSON.parse(event.data)

                    // Handle MCP server response
                    if (data.type === 'mcp_servers_response') {
                        const mcpResponse = data as MCPServerResponse
                        setMcpServers(mcpResponse.servers)
                        setLoadingMcpServers(false)
                        return
                    }

                    // Handle streaming response
                    const streamData = data as StreamResponse
                    if (streamData.choices && streamData.choices[0]) {
                        const choice = streamData.choices[0]

                        if (choice.delta?.content) {
                            console.log('Adding content chunk:', choice.delta.content)
                            setCurrentResponse(prev => prev + choice.delta.content)
                            setIsStreaming(true)
                        }

                        if (choice.finish_reason === 'stop' || choice.finish_reason === 'length') {
                            console.log('Stream finished, finalizing message')
                            setIsStreaming(false)
                            setMessages(prev => [
                                ...prev,
                                {
                                    role: 'assistant',
                                    content: currentResponse + (choice.delta?.content || ''),
                                    timestamp: Date.now()
                                }
                            ])
                            setCurrentResponse('')
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
    }, [url, reconnectAttempts, reconnectInterval, currentResponse])

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

    const sendMessage = useCallback((content: string, model = 'default') => {
        if (connectionStatus !== ConnectionStatus.CONNECTED || !wsRef.current) {
            console.error('WebSocket not connected')
            return
        }

        const userMessage: Message = {
            role: 'user',
            content,
            timestamp: Date.now()
        }

        // Add user message to chat
        setMessages(prev => [...prev, userMessage])

        // Prepare OpenAI-format request
        const request: OpenAIRequest = {
            model,
            messages: [...messages, userMessage],
            stream: true,
            max_tokens: 2048,
            temperature: 0.7
        }

        // Send to WebSocket
        wsRef.current.send(JSON.stringify(request))
        setCurrentResponse('')
        setIsStreaming(true)
    }, [connectionStatus, messages])

    const clearMessages = useCallback(() => {
        setMessages([])
        setCurrentResponse('')
        setIsStreaming(false)
    }, [])

    const getMcpServers = useCallback((query = '') => {
        if (connectionStatus !== ConnectionStatus.CONNECTED || !wsRef.current) {
            console.error('WebSocket not connected')
            return
        }

        setLoadingMcpServers(true)

        const request: MCPServerRequest = {
            type: 'list_mcp_servers',
            query
        }

        wsRef.current.send(JSON.stringify(request))
    }, [connectionStatus])

    useEffect(() => {
        if (autoConnect) {
            connect()
        }

        return () => {
            disconnect()
        }
    }, [autoConnect, connect, disconnect])

    return {
        connectionStatus,
        messages,
        currentResponse,
        isStreaming,
        mcpServers,
        loadingMcpServers,
        connect,
        disconnect,
        sendMessage,
        clearMessages,
        getMcpServers
    }
} 