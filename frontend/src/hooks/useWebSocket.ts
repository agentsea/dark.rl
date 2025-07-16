import { useState, useEffect, useRef, useCallback } from 'react'
import { flushSync } from 'react-dom'
import React from 'react'

export interface Message {
    role: 'user' | 'assistant' | 'system' | 'tool'
    content: string
    timestamp?: string
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
    const [availableTools, setAvailableTools] = useState<MCPServerAction[]>([])

    // Dual response state
    const [isDualResponse, setIsDualResponse] = useState(false);
    const [isDualTransitioning, setIsDualTransitioning] = useState(false);
    const [localResponse, setLocalResponse] = useState('');
    const [gptResponse, setGptResponse] = useState('');
    const [localFinished, setLocalFinished] = useState(false);
    const [gptFinished, setGptFinished] = useState(false);
    const [dualSessionId, setDualSessionId] = useState<number>(0);
    const [localModel, setLocalModel] = useState('');
    const [gptModel, setGptModel] = useState('');
    const [toolResultMessage, setToolResultMessage] = useState<Message | null>(null)

    // Refs for real-time response accumulation
    const localResponseRef = useRef('')
    const gptResponseRef = useRef('')
    const currentSessionIdRef = useRef<number | null>(null)
    const dualInitializedRef = useRef(false)

    // Track which sessions we've already initialized to prevent multiple initializations
    const initializedSessionsRef = useRef(new Set<number>())

    // State for task messages, to be managed by the hook
    const [taskMessages, setTaskMessages] = useState<Message[]>([])

    // DEBUG: Track how many times the emergency fix is applied
    const emergencyFixCountRef = useRef<number>(0)

    // BYPASS BROKEN REACT STATE: Use refs to track dual session state
    const dualSessionIdRef = useRef<number>(0)
    const isDualResponseRef = useRef<boolean>(false)

    // Proper force update mechanism using state counter
    const [forceUpdateCounter, setForceUpdateCounter] = useState(0)
    const forceUpdate = useCallback(() => {
        console.log('🔄 FORCE UPDATE called - current counter:', forceUpdateCounter)
        console.log('🔄   - isDualResponseRef.current:', isDualResponseRef.current)
        console.log('🔄   - dualSessionIdRef.current:', dualSessionIdRef.current)
        setForceUpdateCounter(prev => {
            console.log('🔄   - incrementing counter from', prev, 'to', prev + 1)
            return prev + 1
        })
    }, [])

    // Debug logging when forceUpdateCounter changes
    useEffect(() => {
        console.log('🔄 forceUpdateCounter EFFECT triggered with counter:', forceUpdateCounter)
        console.log('🔄   - isDualResponseRef.current:', isDualResponseRef.current)
        console.log('🔄   - dualSessionIdRef.current:', dualSessionIdRef.current)
    }, [forceUpdateCounter])

    // DEBUG: Log state changes
    useEffect(() => {
        console.log('📊 STATE CHANGE: dualSessionId updated to:', dualSessionId)
    }, [dualSessionId])

    useEffect(() => {
        console.log('📊 STATE CHANGE: isDualResponse updated to:', isDualResponse)
    }, [isDualResponse])

    // DEBUG: Log dual response content state changes
    useEffect(() => {
        console.log('📊 STATE CHANGE: localResponse updated to length:', localResponse.length)
        if (localResponse.length > 0) {
            console.log('📊   - localResponse content preview:', localResponse.substring(0, 100) + '...')
        }
    }, [localResponse])

    useEffect(() => {
        console.log('📊 STATE CHANGE: gptResponse updated to length:', gptResponse.length)
        if (gptResponse.length > 0) {
            console.log('📊   - gptResponse content preview:', gptResponse.substring(0, 100) + '...')
        }
    }, [gptResponse])

    const wsRef = useRef<WebSocket | null>(null)
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    const reconnectCountRef = useRef(0)
    const currentResponseRef = useRef('')

    const connect = useCallback(() => {
        // Prevent multiple connections
        if (wsRef.current) {
            if (wsRef.current.readyState === WebSocket.OPEN) {
                console.log('🔥 WebSocket already connected, skipping connect attempt')
                return
            }
            if (wsRef.current.readyState === WebSocket.CONNECTING) {
                console.log('🔥 WebSocket already connecting, skipping connect attempt')
                return
            }
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
                // RAW MESSAGE DEBUG
                console.log('🔥🔥🔥 RAW WEBSOCKET MESSAGE:', event.data)

                try {
                    const data = JSON.parse(event.data)

                    // MESSAGE TYPE DEBUG
                    console.log('🔥🔥🔥 PARSED MESSAGE TYPE:', data.type)

                    // SPECIAL ALERT FOR DUAL_RESPONSE_START
                    if (data.type === 'dual_response_start') {
                        console.log('🚨🚨🚨 DUAL_RESPONSE_START DETECTED!!! 🚨🚨🚨')
                        console.log('🚨🚨🚨 THIS SHOULD BE THE FIRST MESSAGE FOR DUAL RESPONSE 🚨🚨🚨')
                        console.log('🚨   - session_id:', data.session_id)
                        console.log('🚨   - local_model:', data.local_model)
                        console.log('🚨   - gpt_model:', data.gpt_model)
                        console.log('🚨   - is_continuation:', data.is_continuation)
                        console.log('🚨   - current dualSessionId state:', dualSessionId)
                        console.log('🚨   - current isDualResponse state:', isDualResponse)
                    }

                    // DEBUG: Log all message types to see what we're actually receiving
                    if (data.type !== 'dual_response_chunk') {
                        console.log('📬 NON-CHUNK MESSAGE:', data.type, 'session_id:', data.session_id)
                    }

                    // Handle task updates pushed from the server
                    if (data.type === 'task_updated') {
                        console.log('🔄 Received task_updated, updating messages...')
                        setTaskMessages(data.messages)
                        return
                    }

                    // Handle MCP server actions response
                    if (data.type === 'mcp_server_actions_response') {
                        const actionsResponse = data as MCPServerActionsResponse
                        setMcpServerActions(prev => ({
                            ...prev,
                            [actionsResponse.server_id]: actionsResponse.actions
                        }))

                        // Also update a flat list of available tools for the correction modal
                        setAvailableTools(prev => {
                            const newActions = actionsResponse.actions.map(action => ({
                                ...action,
                                // Prepend server_id to name to make it unique
                                name: `${actionsResponse.server_id}.${action.name}`
                            }))
                            // Avoid duplicates
                            const existingNames = new Set(prev.map(a => a.name))
                            const filteredNewActions = newActions.filter(a => !existingNames.has(a.name))
                            return [...prev, ...filteredNewActions]
                        })

                        setLoadingMcpServerActions(false)
                        return
                    }

                    // Log all message types we receive
                    if (data.type !== 'dual_response_chunk') {
                        console.log('📢📢📢 NON-CHUNK MESSAGE RECEIVED:', data.type, data)
                    }

                    // SPECIAL DEBUG FOR DUAL_RESPONSE_START
                    if (data.type === 'dual_response_start') {
                        console.log('🔥 DUAL_RESPONSE_START RECEIVED:', data)
                        console.log('🔥   - session_id:', data.session_id)
                        console.log('🔥   - local_model:', data.local_model)
                        console.log('🔥   - gpt_model:', data.gpt_model)
                        console.log('🔥   - is_continuation:', data.is_continuation)
                        console.log('🔥   - Current dualSessionId state before:', dualSessionId)
                        console.log('🔥   - Current dualInitializedRef before:', dualInitializedRef.current)
                        console.log('🔥   - initializedSessionsRef.current has session:', initializedSessionsRef.current.has(data.session_id))

                        // Mark this session as initialized immediately
                        initializedSessionsRef.current.add(data.session_id)
                        dualInitializedRef.current = true
                        currentSessionIdRef.current = data.session_id

                        // BYPASS BROKEN REACT STATE: Use refs AND state
                        dualSessionIdRef.current = data.session_id
                        isDualResponseRef.current = true
                        forceUpdate() // Force re-render when refs change

                        // AGGRESSIVE state updates to overcome React batching
                        // If this is a new session, reset the previous state
                        if (dualSessionId !== 0 && dualSessionId !== data.session_id) {
                            console.log('🔥   - New session detected, resetting previous session state')
                            console.log('🔥   - Previous session:', dualSessionId, '-> New session:', data.session_id)
                        }

                        setDualSessionId(data.session_id)
                        setIsDualResponse(true)
                        setIsStreaming(true)

                        // Reset response state - always reset for new session
                        setLocalResponse('')
                        setGptResponse('')
                        localResponseRef.current = ''
                        gptResponseRef.current = ''
                        setLocalFinished(false)
                        setGptFinished(false)

                        // Set model names from the dual response start message
                        setLocalModel(data.local_model || 'qwen2.5-vl')
                        setGptModel(data.gpt_model || 'gpt-4.1')

                        console.log('🔥 DUAL_RESPONSE_START: Set refs and state')
                        console.log('🔥   - dualSessionIdRef.current:', dualSessionIdRef.current)
                        console.log('🔥   - isDualResponseRef.current:', isDualResponseRef.current)
                        console.log('🔥   - dualSessionId state:', dualSessionId)
                        console.log('🔥   - isDualResponse state:', isDualResponse)

                        // Force re-render to ensure UI updates
                        forceUpdate()

                        console.log('🔥 DUAL_RESPONSE_START: Handler completed, marked session as initialized')
                        return
                    }

                    // FALLBACK: If we receive dual_response_chunk but haven't initialized dual mode, do it now
                    if (data.type === 'dual_response_chunk') {
                        const chunkSessionId = data.session_id

                        console.log('🔥 RAW CHUNK RECEIVED:')
                        console.log('🔥   - source:', data.source)
                        console.log('🔥   - content:', data.choices?.[0]?.delta?.content)
                        console.log('🔥   - finish_reason:', data.choices?.[0]?.finish_reason)
                        console.log('🔥   - chunkSessionId:', chunkSessionId)
                        console.log('🔥   - dualSessionId (state):', dualSessionId)
                        console.log('🔥   - currentSessionIdRef.current:', currentSessionIdRef.current)
                        console.log('🔥   - dualInitializedRef.current:', dualInitializedRef.current)
                        console.log('🔥   - isDualResponse (state):', isDualResponse)
                        console.log('🔥   - isStreaming (state):', isStreaming)
                        console.log('🔥   - emergencyFixCountRef.current:', emergencyFixCountRef.current)

                        // IMPROVED FALLBACK: Only initialize if we haven't for this specific session
                        // The key issue is that dualSessionId might still be 0 from previous session reset
                        const needsInitialization = chunkSessionId && (
                            !initializedSessionsRef.current.has(chunkSessionId) &&
                            (dualSessionId === 0 || dualSessionId !== chunkSessionId)
                        )

                        // WARNING: Detect missing dual_response_start message
                        if (needsInitialization && chunkSessionId) {
                            console.log('🚨 WARNING: Received dual_response_chunk without dual_response_start!')
                            console.log('🚨   - This suggests the dual_response_start message was lost or not received')
                            console.log('🚨   - Server session ID:', chunkSessionId)
                            console.log('🚨   - Falling back to emergency initialization')
                        }

                        console.log('🔥 FALLBACK CHECK:')
                        console.log('🔥   - chunkSessionId:', chunkSessionId)
                        console.log('🔥   - initializedSessionsRef.current has session:', initializedSessionsRef.current.has(chunkSessionId))
                        console.log('🔥   - dualSessionId:', dualSessionId)
                        console.log('🔥   - dualInitializedRef.current:', dualInitializedRef.current)
                        console.log('🔥   - needsInitialization:', needsInitialization)

                        if (needsInitialization) {
                            console.log('🔥 FALLBACK: Initializing dual mode from first chunk')
                            console.log('🔥   - Reason: Session not initialized or dualSessionId mismatch')
                            console.log('🔥   - Setting dualSessionId to:', chunkSessionId)

                            // Mark this session as initialized
                            initializedSessionsRef.current.add(chunkSessionId)
                            dualInitializedRef.current = true
                            currentSessionIdRef.current = chunkSessionId

                            // BYPASS BROKEN REACT STATE: Set refs directly first
                            dualSessionIdRef.current = chunkSessionId
                            isDualResponseRef.current = true
                            forceUpdate() // Force re-render when refs change

                            // Also try to set React state (but don't rely on it)
                            setDualSessionId(chunkSessionId)
                            setIsDualResponse(true)
                            setIsStreaming(true)

                            // Always reset response state for new session
                            setLocalResponse('')
                            setGptResponse('')
                            localResponseRef.current = ''
                            gptResponseRef.current = ''
                            setLocalFinished(false)
                            setGptFinished(false)

                            // Force re-render to ensure UI updates
                            forceUpdate()

                            console.log('🔥 FALLBACK: Dual mode initialized successfully')
                            console.log('🔥   - dualSessionIdRef.current:', dualSessionIdRef.current)
                            console.log('🔥   - isDualResponseRef.current:', isDualResponseRef.current)
                        }

                        // ENSURE REFS ARE CORRECT: Always sync refs with session ID (bypass broken React state)
                        if (chunkSessionId && initializedSessionsRef.current.has(chunkSessionId)) {
                            // Always ensure refs are correct for valid sessions
                            if (dualSessionIdRef.current !== chunkSessionId) {
                                console.log('🔥 SYNCING REFS: dualSessionIdRef.current', dualSessionIdRef.current, '->', chunkSessionId)
                                dualSessionIdRef.current = chunkSessionId
                                isDualResponseRef.current = true
                                // Force re-render to ensure UI updates
                                forceUpdate()
                            }
                        }

                        // Skip processing if we don't have a valid session match
                        if (!chunkSessionId || (currentSessionIdRef.current && chunkSessionId !== currentSessionIdRef.current)) {
                            console.log('🔥 SKIPPING CHUNK - session mismatch or invalid:', chunkSessionId, 'vs current:', currentSessionIdRef.current)
                            return
                        }

                        console.log('🔥 PROCESSING CHUNK:')
                        console.log('🔥   - source:', data.source)
                        console.log('🔥   - content:', data.choices?.[0]?.delta?.content)
                        console.log('🔥   - finish_reason:', data.choices?.[0]?.finish_reason)

                        const content = data.choices?.[0]?.delta?.content || ''
                        const finishReason = data.choices?.[0]?.finish_reason

                        if (data.source === 'local') {
                            const prevLength = localResponseRef.current.length
                            localResponseRef.current += content
                            const newLength = localResponseRef.current.length
                            console.log('🔥   - localResponseRef updated: ', prevLength, ' -> ', newLength)
                            console.log('🔥   - Added content: "', content, '"')

                            // CRITICAL: Update React state so UI can render the content
                            console.log('🔥   - About to call setLocalResponse with length:', localResponseRef.current.length)
                            setLocalResponse(localResponseRef.current)
                            console.log('🔥   - Called setLocalResponse - React should update localResponse state')

                            if (finishReason === 'stop') {
                                console.log('🔥 LOCAL RESPONSE FINISHED with reason:', finishReason)
                                console.log('🔥   - Setting localFinished to TRUE')
                                console.log('🔥   - Final localResponse length:', localResponseRef.current.length)
                                setLocalFinished(true)
                            }
                        } else if (data.source === 'gpt') {
                            const prevLength = gptResponseRef.current.length
                            gptResponseRef.current += content
                            const newLength = gptResponseRef.current.length
                            console.log('🔥   - gptResponseRef updated: ', prevLength, ' -> ', newLength)
                            console.log('🔥   - Added content: "', content, '"')

                            // CRITICAL: Update React state so UI can render the content
                            console.log('🔥   - About to call setGptResponse with length:', gptResponseRef.current.length)
                            setGptResponse(gptResponseRef.current)
                            console.log('🔥   - Called setGptResponse - React should update gptResponse state')

                            if (finishReason === 'stop') {
                                console.log('🔥 GPT RESPONSE FINISHED with reason:', finishReason)
                                console.log('🔥   - Setting gptFinished to TRUE')
                                console.log('🔥   - Final gptResponse length:', gptResponseRef.current.length)
                                setGptFinished(true)
                            }
                        }

                        console.log('Stream finished, finalizing message')
                        return
                    }

                    if (data.type === 'dual_response_complete') {
                        console.log('🔥 DUAL RESPONSE COMPLETE RECEIVED:', data)

                        // Check if this completion is for the current session
                        if (data.session_id && data.session_id !== currentSessionIdRef.current) {
                            console.log('🔥 IGNORING OLD COMPLETION: session', data.session_id, 'vs current', currentSessionIdRef.current)
                            return
                        }

                        console.log('🔥 Session comparison - completion:', data.session_id, 'vs current:', currentSessionIdRef.current)

                        // Force sync state with refs and finish
                        console.log('🔥 DUAL RESPONSE COMPLETE DEBUG:')
                        console.log('🔥   - localResponse state length:', localResponse.length)
                        console.log('🔥   - localResponseRef length:', localResponseRef.current.length)
                        console.log('🔥   - gptResponse state length:', gptResponse.length)
                        console.log('🔥   - gptResponseRef length:', gptResponseRef.current.length)
                        console.log('🔥   - localFinished before:', localFinished)
                        console.log('🔥   - gptFinished before:', gptFinished)
                        console.log('🔥   - backend reports local_finished:', data.local_finished)
                        console.log('🔥   - backend reports gpt_finished:', data.gpt_finished)
                        console.log('🔥   - session_id:', data.session_id)

                        // Sync state with refs
                        if (localResponseRef.current.length > 0 && localResponse.length === 0) {
                            console.log('🔥   - Syncing localResponse state with ref')
                            setLocalResponse(localResponseRef.current)
                        }
                        if (gptResponseRef.current.length > 0 && gptResponse.length === 0) {
                            console.log('🔥   - Syncing gptResponse state with ref')
                            setGptResponse(gptResponseRef.current)
                        }

                        // Force final state updates
                        console.log('🔥   - Syncing localResponse state with ref')
                        setLocalResponse(localResponseRef.current)

                        console.log('🔥   - Syncing gptResponse state with ref')
                        setGptResponse(gptResponseRef.current)

                        // Force final state updates
                        console.log('🔥   - Force setting localFinished to:', data.local_finished)
                        setLocalFinished(data.local_finished)

                        console.log('🔥   - Force setting gptFinished to:', data.gpt_finished)
                        setGptFinished(data.gpt_finished)

                        // End streaming and reset initialization flag
                        console.log('🔥   - Set isStreaming to false')
                        setIsStreaming(false)

                        // IMPORTANT: Don't reset dualInitializedRef immediately to avoid race conditions
                        // Let it reset naturally when the next session starts
                        console.log('🔥   - Keeping dualInitializedRef.current as true for now')

                        // CRITICAL FIX: Don't reset dualSessionId immediately! Keep it so the UI stays visible
                        // The dualSessionId should only be reset when:
                        // 1. A new dual response starts (with different session ID)
                        // 2. User makes a selection and next response begins
                        console.log('🔥   - KEEPING dualSessionId for user selection:', data.session_id)
                        console.log('🔥   - UI should remain visible for user to select between responses')

                        // Clean up session tracking to prevent memory leaks
                        console.log('🔥   - Cleaning up session tracking for completed session:', data.session_id)
                        initializedSessionsRef.current.delete(data.session_id)

                        // DON'T reset dualSessionId or isDualResponse here - let the UI stay visible
                        // The reset will happen when the user makes a selection or a new dual response starts

                        console.log('🔥   - Dual response completion processing done - UI should stay visible')

                        // Add a minimum display time to prevent immediate clearing
                        console.log('🔥   - Setting 3-second minimum display time for dual response')
                        setTimeout(() => {
                            console.log('🔥   - Minimum display time expired - dual response can now be cleared by user selection')
                        }, 3000)

                        return
                    }

                    // Handle MCP server response
                    if (data.type === 'mcp_servers_response') {
                        const mcpResponse = data as MCPServerResponse
                        setMcpServers(mcpResponse.servers)
                        setLoadingMcpServers(false)
                        return
                    }

                    // Handle model selection response - reset dual state
                    if (data.type === 'model_selection_response') {
                        if (data.success) {
                            console.log('🔥 MODEL SELECTION: User selected model, resetting dual state')
                            console.log('🔥   - Selected model:', data.selected_model)
                            console.log('🔥   - Resetting dualSessionId to 0')
                            setDualSessionId(0)
                            console.log('📊 setIsDualResponse(false) called from model_selection_response')
                            console.log('📊 WHO CALLED?', new Error().stack)
                            setIsDualResponse(false)
                            setIsStreaming(false)
                            dualInitializedRef.current = false
                            setToolResultMessage(null) // Reset tool result message on model selection

                            // Reset refs and force update
                            dualSessionIdRef.current = 0
                            isDualResponseRef.current = false
                            forceUpdate()
                        }
                        return
                    }

                    // Handle streaming response (single model)
                    const streamData = data as StreamResponse
                    if (streamData.choices && streamData.choices[0]) {
                        const choice = streamData.choices[0]

                        if (choice.delta?.content) {
                            // Reset dual state when single model streaming starts
                            if (dualSessionId !== 0) {
                                console.log('🔥 SINGLE MODEL: Resetting dual state for single model streaming')
                                setDualSessionId(0)
                                console.log('📊 setIsDualResponse(false) called from single model streaming')
                                console.log('📊 WHO CALLED?', new Error().stack)
                                setIsDualResponse(false)
                                dualInitializedRef.current = false
                            }

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
                                    timestamp: new Date().toISOString()
                                }
                            ])

                            // Clear the response and reset for next message
                            currentResponseRef.current = ''
                            // Keep currentResponse visible until user continues conversation
                        }
                    }

                    if (data.type === 'tool_result') {
                        console.log('✅ Received tool result, adding to messages:', data.message)
                        setToolResultMessage(data.message)
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error, 'Raw data:', event.data)
                    setConnectionStatus(ConnectionStatus.ERROR)
                }
            }

            wsRef.current.onclose = (event) => {
                setConnectionStatus(ConnectionStatus.DISCONNECTED)
                setIsStreaming(false)
                setToolResultMessage(null) // Reset tool result message on disconnect
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
        setToolResultMessage(null) // Reset tool result message on disconnect
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
            timestamp: new Date().toISOString()
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
        console.log(`🔥 WEBSOCKET SEND: ${JSON.stringify(request).substring(0, 200)}...`)
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
        console.log('🔥 WHO CALLED ME?', new Error().stack)
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

        // Clear refs immediately
        localResponseRef.current = '';
        gptResponseRef.current = '';
        currentSessionIdRef.current = 0; // Reset session ref too
        dualInitializedRef.current = false; // Reset initialization flag

        // Clear dual response refs
        dualSessionIdRef.current = 0;
        isDualResponseRef.current = false;
        forceUpdate(); // Force re-render when refs are cleared

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
        setToolResultMessage(null) // Reset tool result message on clearDualResponse

        console.log('🔥 State after clearing:')
        console.log('🔥   - isDualResponse:', false)
        console.log('🔥   - localResponseRef.current length:', localResponseRef.current.length)
        console.log('🔥   - gptResponseRef.current length:', gptResponseRef.current.length)
    }, [isDualResponse, localResponse.length, gptResponse.length, localModel, gptModel, localFinished, gptFinished, localResponseRef, gptResponseRef]);

    const selectModel = useCallback(async (selectedModel: 'local' | 'gpt', taskId?: string, editedContent?: string) => {
        console.log(`🐛 [useWebSocket] selectModel called with model=${selectedModel}, taskId=${taskId}`)

        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('🐛 [useWebSocket] WebSocket not connected for model selection')
            return
        }

        // 🔥 CRITICAL FIX: Capture current responses BEFORE clearing
        const currentLocalResponse = editedContent && selectedModel === 'local' ? editedContent : localResponseRef.current
        const currentGptResponse = editedContent && selectedModel === 'gpt' ? editedContent : gptResponseRef.current

        // 🔥 CRITICAL FIX: Clear content IMMEDIATELY when user selects to prevent flash
        console.log('🔥 IMMEDIATE CLEAR: Clearing all content before sending request')
        clearDualResponse()

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
            console.log(`🔥 WEBSOCKET SEND (selectModel): ${JSON.stringify(request).substring(0, 200)}...`)

            // Create a one-time listener for the model selection response
            const handleModelSelectionResponse = (event: MessageEvent) => {
                try {
                    const data = JSON.parse(event.data)
                    console.log(`🐛 [useWebSocket] Received message in selectModel:`, data.type)

                    if (data.type === 'model_selection_response') {
                        console.log(`🐛 [useWebSocket] Model selection response received, success=${data.success}`)
                        // Remove this listener
                        if (wsRef.current) {
                            wsRef.current.removeEventListener('message', handleModelSelectionResponse)
                        }
                        resolve(data)
                        return
                    }
                } catch (error) {
                    console.error('🐛 [useWebSocket] Error parsing selectModel response:', error)
                }
            }

            // Add the listener
            if (wsRef.current) {
                wsRef.current.addEventListener('message', handleModelSelectionResponse)
            }

            // Send the request
            if (wsRef.current) {
                wsRef.current.send(JSON.stringify(request))
                console.log(`🐛 [useWebSocket] Model selection request sent`)
            } else {
                reject(new Error('WebSocket not connected for model selection'))
                return
            }

            // Set timeout for response
            setTimeout(() => {
                if (wsRef.current) {
                    wsRef.current.removeEventListener('message', handleModelSelectionResponse)
                }
                reject(new Error('Model selection timeout'))
            }, 10000) // Increased timeout to 10 seconds
        })
    }, [localModel, gptModel, clearDualResponse])

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
            console.error('�� [useWebSocket] WebSocket not connected for task retrieval after waiting')
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
                        // Also update the local task messages state
                        setTaskMessages(data.messages || [])
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

    const sendDualResponseComment = useCallback(async (
        taskId: string,
        messageIndex: number,
        comment: string
    ): Promise<any> => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected for dual response comment')
            return null
        }

        return new Promise((resolve) => {
            const request = {
                type: 'dual_response_comment',
                task_id: taskId,
                message_index: messageIndex,
                comment: comment
            }

            // Set up one-time listener for dual response comment response
            const ws = wsRef.current!
            const originalOnMessage = ws.onmessage

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)

                    if (data.type === 'dual_response_comment_processed') {
                        // Restore original message handler
                        if (wsRef.current) {
                            wsRef.current.onmessage = originalOnMessage
                        }
                        resolve(data)
                        return
                    }

                    if (data.type === 'error') {
                        console.error('Dual response comment error:', data.error)
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
                    console.error('Error parsing dual response comment response:', error)
                    if (wsRef.current) {
                        wsRef.current.onmessage = originalOnMessage
                    }
                    resolve({ error: 'Failed to parse response' })
                }
            }

            ws.send(JSON.stringify(request))
        })
    }, [connectionStatus])

    // DEBUG: Log the return values to verify refs are working
    const returnValues = {
        connectionStatus,
        messages,
        taskMessages, // Expose task messages
        currentResponse,
        isStreaming,
        mcpServers,
        loadingMcpServers,
        mcpServerActions,
        loadingMcpServerActions,
        availableTools,
        // Dual response state - USE REFS for reliable values
        isDualResponse: isDualResponseRef.current, // Use ref value directly - no transition hiding
        isDualTransitioning,
        localResponse,
        gptResponse,
        localModel,
        gptModel,
        localFinished,
        gptFinished,
        dualSessionId: dualSessionIdRef.current, // Use ref value instead of broken state
        currentSessionIdRef,
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
        sendCorrectionWithExecution,
        sendDualResponseComment,
        // CRITICAL: Include forceUpdateCounter to ensure re-renders happen
        _forceUpdateCounter: forceUpdateCounter,
        toolResultMessage
    }

    // DEBUG: Log when dual response values change
    if (isDualResponseRef.current || dualSessionIdRef.current !== 0) {
        console.log('🚀 RETURN VALUES DEBUG:')
        console.log('🚀   - forceUpdateCounter:', forceUpdateCounter)
        console.log('🚀   - isDualResponseRef.current:', isDualResponseRef.current)
        console.log('🚀   - dualSessionIdRef.current:', dualSessionIdRef.current)
        console.log('🚀   - isDualTransitioning:', isDualTransitioning)
        console.log('🚀   - isDualResponse calculation:', isDualResponseRef.current && !isDualTransitioning)
        console.log('🚀   - isDualResponse (returned):', returnValues.isDualResponse)
        console.log('🚀   - dualSessionId (returned):', returnValues.dualSessionId)
        console.log('🚀   - localResponse length:', returnValues.localResponse.length)
        console.log('🚀   - gptResponse length:', returnValues.gptResponse.length)
        console.log('🚀   - localFinished:', returnValues.localFinished)
        console.log('🚀   - gptFinished:', returnValues.gptFinished)
        console.log('🚀   - toolResultMessage:', returnValues.toolResultMessage)
    }

    return returnValues
} 