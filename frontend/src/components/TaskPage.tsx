import React, { useState, useEffect, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useParams, useNavigate } from 'react-router-dom'
import TypingAnimation from './TypingAnimation'
import MCPServerDropdown from './MCPServerDropdown'
import CorrectionModal from './CorrectionModal'
import useWebSocket, { ConnectionStatus } from '../hooks/useWebSocket'
import type { Message } from '../hooks/useWebSocket'

interface Task {
    id: string
    title: string
    initial_prompt: string
    model: string
    mcp_servers: string[]
    created_at: string
    updated_at: string
    status: string
    current_state: string
}

interface TaskMessage extends Message {
    timestamp: string;
}

interface PendingDualResponse {
    task_id: string
    local_response: string
    gpt_response: string
    local_model: string
    gpt_model: string
    local_finished: boolean
    gpt_finished: boolean
    session_id: number
}

// Component to display tool responses with collapsible JSON
function ToolResponseDisplay({ content }: { content: string }): React.JSX.Element {
    const [isExpanded, setIsExpanded] = useState(false)

    try {
        const jsonData = JSON.parse(content)
        const isSuccess = jsonData.success === true

        return (
            <div style={{
                backgroundColor: isSuccess ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                border: `1px solid ${isSuccess ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                borderRadius: '8px',
                padding: '12px',
                marginTop: '8px',
                marginBottom: '8px',
                maxWidth: '600px'
            }}>
                {/* Status indicator and toggle button */}
                <div
                    className="flex items-center justify-between cursor-pointer"
                    onClick={() => setIsExpanded(!isExpanded)}
                >
                    <div className="flex items-center gap-2">
                        <span style={{
                            color: isSuccess ? '#22c55e' : '#ef4444',
                            fontSize: '14px',
                            fontWeight: 'bold'
                        }}>
                            {isSuccess ? '✅ Tool Executed Successfully' : '❌ Tool Execution Failed'}
                        </span>
                        {jsonData.content && (
                            <span style={{ color: '#9CA3AF', fontSize: '12px' }}>
                                ({jsonData.content.length} chars)
                            </span>
                        )}
                    </div>
                    <span style={{
                        color: '#9CA3AF',
                        fontSize: '12px',
                        transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                        transition: 'transform 0.2s'
                    }}>
                        ▼
                    </span>
                </div>

                {/* Expandable JSON content */}
                {isExpanded && (
                    <div style={{ marginTop: '12px' }}>
                        <pre style={{
                            backgroundColor: 'rgba(0, 0, 0, 0.3)',
                            border: '1px solid rgba(107, 114, 128, 0.3)',
                            borderRadius: '4px',
                            padding: '12px',
                            fontSize: '12px',
                            fontFamily: 'monospace',
                            color: '#E5E7EB',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                            overflow: 'auto',
                            maxHeight: '400px'
                        }}>
                            {JSON.stringify(jsonData, null, 2)}
                        </pre>
                    </div>
                )}
            </div>
        )
    } catch (error) {
        // Fallback for non-JSON content
        return (
            <div style={{
                backgroundColor: 'rgba(107, 114, 128, 0.1)',
                border: '1px solid rgba(107, 114, 128, 0.3)',
                borderRadius: '8px',
                padding: '12px',
                marginTop: '8px',
                marginBottom: '8px',
                fontSize: '12px',
                fontFamily: 'monospace',
                color: '#E5E7EB',
                maxWidth: '600px'
            }}>
                {content}
            </div>
        )
    }
}

// Helper function to parse and render content with special tags
function parseContent(content: string): React.JSX.Element {
    const parts: React.JSX.Element[] = []
    let currentIndex = 0

    // Find all <think>, <tool_call>, and <tool_response> tags
    const thinkRegex = /<think>(.*?)<\/think>/gs
    const toolCallRegex = /<tool_call>(.*?)<\/tool_call>/gs
    const toolResponseRegex = /<tool_response>(.*?)<\/tool_response>/gs

    // Combine all patterns and find all matches with their positions
    const allMatches: Array<{ match: RegExpExecArray; type: 'think' | 'tool_call' | 'tool_response' }> = []

    let match
    while ((match = thinkRegex.exec(content)) !== null) {
        allMatches.push({ match, type: 'think' })
    }

    while ((match = toolCallRegex.exec(content)) !== null) {
        allMatches.push({ match, type: 'tool_call' })
    }

    while ((match = toolResponseRegex.exec(content)) !== null) {
        allMatches.push({ match, type: 'tool_response' })
    }

    // Sort matches by position
    allMatches.sort((a, b) => a.match.index - b.match.index)

    // Process each match
    allMatches.forEach((matchObj, index) => {
        const { match, type } = matchObj

        // Add text before this match
        if (match.index > currentIndex) {
            const beforeText = content.slice(currentIndex, match.index)
            if (beforeText) {
                parts.push(<span key={`text-${index}`}>{beforeText}</span>)
            }
        }

        // Add the special tag content
        if (type === 'think') {
            parts.push(
                <div key={`think-${index}`} style={{
                    color: '#9CA3AF',
                    fontSize: '0.88rem',
                    fontStyle: 'italic',
                    marginTop: '8px',
                    marginBottom: '8px',
                    paddingLeft: '16px',
                    borderLeft: '2px solid #4B5563'
                }}>
                    💭 {match[1]}
                </div>
            )
        } else if (type === 'tool_call') {
            try {
                const toolCallData = JSON.parse(match[1])
                parts.push(
                    <div key={`tool-${index}`} style={{
                        backgroundColor: 'rgba(97, 253, 252, 0.1)',
                        border: '1px solid rgba(97, 253, 252, 0.3)',
                        borderRadius: '8px',
                        padding: '12px',
                        marginTop: '8px',
                        marginBottom: '8px',
                        fontSize: '0.9em',
                        maxWidth: '600px'
                    }}>
                        <div style={{ color: '#61FDFC', fontWeight: 'bold', marginBottom: '4px' }}>
                            🔧 Tool Call
                        </div>
                        <pre style={{
                            color: '#E5E7EB',
                            fontSize: '0.85em',
                            fontFamily: 'monospace',
                            whiteSpace: 'pre-wrap',
                            margin: 0
                        }}>
                            {JSON.stringify(toolCallData, null, 2)}
                        </pre>
                    </div>
                )
            } catch (e) {
                // If JSON parsing fails, show raw content
                parts.push(
                    <div key={`tool-${index}`} style={{
                        backgroundColor: 'rgba(97, 253, 252, 0.1)',
                        border: '1px solid rgba(97, 253, 252, 0.3)',
                        borderRadius: '8px',
                        padding: '12px',
                        marginTop: '8px',
                        marginBottom: '8px',
                        fontSize: '0.9em',
                        maxWidth: '600px'
                    }}>
                        <div style={{ color: '#61FDFC', fontWeight: 'bold', marginBottom: '4px' }}>
                            🔧 Tool Call
                        </div>
                        <div style={{ color: '#E5E7EB', fontSize: '0.85em' }}>
                            {match[1]}
                        </div>
                    </div>
                )
            }
        } else if (type === 'tool_response') {
            // Use the same ToolResponseDisplay component for tool responses
            parts.push(
                <div key={`tool-response-${index}`} style={{ marginTop: '8px', marginBottom: '8px' }}>
                    <ToolResponseDisplay content={match[1]} />
                </div>
            )
        }

        currentIndex = match.index + match[0].length
    })

    // Add any remaining text after the last match
    if (currentIndex < content.length) {
        const remainingText = content.slice(currentIndex)
        if (remainingText) {
            parts.push(<span key="remaining">{remainingText}</span>)
        }
    }

    // If no special tags found, return the original content
    if (parts.length === 0) {
        return <span>{content}</span>
    }

    return <>{parts}</>
}

function TaskPage() {
    const { id } = useParams<{ id: string }>()
    const navigate = useNavigate()
    const [task, setTask] = useState<Task | null>(null)
    const [taskMessages, setTaskMessages] = useState<TaskMessage[]>([])
    const [pendingDualResponses, setPendingDualResponses] = useState<PendingDualResponse | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [userInput, setUserInput] = useState('')
    const bottomRef = useRef<HTMLDivElement>(null)

    // Auto-scroll to bottom function
    const scrollToBottom = () => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    const {
        connectionStatus,
        currentResponse,
        isStreaming,
        isDualResponse,
        dualSessionId,
        localResponse,
        gptResponse,
        localFinished,
        gptFinished,
        toolResultMessage,
        connect,
        sendMessage,
        sendLearningFeedback,
        sendCorrectionWithExecution,
        getTask,
        selectModel,
        sendDualResponseComment,
        clearDualResponse
    } = useWebSocket({
        url: 'ws://localhost:8000',
        autoConnect: true,
        reconnectAttempts: 3,
        reconnectInterval: 3000
    })

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!userInput.trim() || !id || !task) return

        if (isDualResponse) {
            // If it's a dual response, treat this as a clarifying comment
            await sendDualResponseComment(id, taskMessages.length, userInput)
            clearDualResponse()
        } else {
            // Otherwise, it's a new user message in the conversation
            sendMessage(userInput, task.model, [...taskMessages, { role: 'user', content: userInput, timestamp: new Date().toISOString() }], id, false)
        }

        // Add the message to the local state for immediate UI update
        setTaskMessages(prev => [
            ...prev,
            {
                role: 'user',
                content: userInput,
                timestamp: new Date().toISOString()
            }
        ])

        // Reset input
        setUserInput('')
    }


    // Load task data from server using the useWebSocket hook
    const loadTask = useCallback(async () => {
        if (!id || connectionStatus !== ConnectionStatus.CONNECTED) {
            return
        }

        try {
            setLoading(true)
            setError(null)

            // Use the getTask function from the hook
            const taskData = await getTask(id)

            if (taskData && !taskData.error) {
                // Update state with server data
                setTask(taskData.task)
                setTaskMessages(taskData.messages || [])
                setPendingDualResponses(taskData.pending_dual)

                // Debug logging for task state
                console.log('🔍 Task data loaded:', {
                    taskId: taskData.task.id,
                    currentState: taskData.task.current_state,
                    messagesCount: taskData.messages?.length || 0,
                    hasPendingDual: !!taskData.pending_dual,
                    pendingDualInfo: taskData.pending_dual ? {
                        localLength: taskData.pending_dual.local_response?.length || 0,
                        gptLength: taskData.pending_dual.gpt_response?.length || 0,
                        localFinished: taskData.pending_dual.local_finished,
                        gptFinished: taskData.pending_dual.gpt_finished,
                        sessionId: taskData.pending_dual.session_id
                    } : null
                })

                // SIMPLIFIED AUTO-TRIGGER: Only trigger in very specific cases
                // 1. New task: exactly 1 user message, no assistant messages
                // 2. After tool response: last message is user with <tool_response> tags, preceded by assistant message
                // 3. After user comment: last message is user without <tool_response> tags, preceded by assistant message

                const messages = taskData.messages
                const lastMessage = messages[messages.length - 1]
                const secondLastMessage = messages.length > 1 ? messages[messages.length - 2] : null

                // Case 1: New task
                const isNewTask = (
                    messages.length === 1 &&
                    messages[0].role === 'user' &&
                    taskData.task.current_state === 'idle'
                )

                // Case 2: After tool response (user message with <tool_response> tags)
                const isAfterToolResponse = (
                    messages.length >= 2 &&
                    lastMessage.role === 'user' &&
                    lastMessage.content.includes('<tool_response>') &&
                    secondLastMessage?.role === 'assistant' &&
                    taskData.task.current_state === 'idle'
                )

                // Case 3: After user comment (user message without <tool_response> tags, following assistant)
                const isAfterUserComment = (
                    messages.length >= 2 &&
                    lastMessage.role === 'user' &&
                    !lastMessage.content.includes('<tool_response>') &&
                    secondLastMessage?.role === 'assistant' &&
                    taskData.task.current_state === 'idle'
                )

                // STRICT safety checks: Don't auto-trigger if:
                // - We already have dual responses in progress
                // - Task state is not 'idle' (includes 'processing', 'streaming_dual', 'awaiting_dual_selection')
                // - We're currently streaming
                // - We have pending dual responses
                // - We have an active dual session (extra safety)
                const hasActiveResponses = isDualResponse || !!taskData.pending_dual ||
                    taskData.task.current_state !== 'idle' ||
                    isStreaming ||
                    (dualSessionId && dualSessionId > 0)

                const shouldAutoTrigger = (isNewTask || isAfterToolResponse || isAfterUserComment) && !hasActiveResponses

                console.log('🔍 SIMPLIFIED Auto-trigger evaluation:', {
                    isNewTask: isNewTask,
                    isAfterToolResponse: isAfterToolResponse,
                    isAfterUserComment: isAfterUserComment,
                    hasActiveResponses: hasActiveResponses,
                    shouldAutoTrigger: shouldAutoTrigger,
                    messagesCount: messages.length,
                    currentState: taskData.task.current_state,
                    lastMessageRole: lastMessage?.role,
                    lastMessageContent: lastMessage?.content.substring(0, 100) + '...',
                    lastMessageHasToolResponse: lastMessage?.content.includes('<tool_response>'),
                    secondLastMessageRole: secondLastMessage?.role,
                    isDualResponseActive: isDualResponse,
                    dualSessionId: dualSessionId,
                    hasPendingDual: !!taskData.pending_dual,
                    isStreaming: isStreaming
                })

                if (shouldAutoTrigger) {
                    let triggerReason = 'unknown'
                    if (isNewTask) triggerReason = 'NEW TASK'
                    else if (isAfterToolResponse) triggerReason = 'AFTER TOOL RESPONSE'
                    else if (isAfterUserComment) triggerReason = 'AFTER USER COMMENT'

                    console.log(`🚀 Auto-triggering AI response for: ${triggerReason}`)

                    // Use auto_response flag to indicate this is an automatic generation
                    // Execute immediately to avoid connection stability issues
                    console.log('🚀 Executing auto-trigger immediately')
                    sendMessage(
                        lastMessage.content, // content: string (last user message)
                        taskData.task.model, // model: string  
                        messages, // customMessages: Message[] (full history)
                        id, // taskId: string
                        true // isAutoResponse: boolean
                    )
                } else {
                    console.log('🔄 Not auto-triggering - conditions not met or has active responses')
                }
            } else {
                setError(taskData?.error || 'Failed to load task')
            }

            setLoading(false)

        } catch (error) {
            console.error('Error loading task:', error)
            setError(error instanceof Error ? error.message : 'Failed to load task')
            setLoading(false)
        }
    }, [id, connectionStatus, getTask, sendMessage])

    // Auto-connect when component mounts
    useEffect(() => {
        connect()
    }, [connect])

    // Load task when component mounts or connection changes
    useEffect(() => {
        loadTask()
    }, [loadTask])

    // Handle model selection - let backend handle the complete flow
    const handleModelSelection = async (selectedModel: 'local' | 'gpt') => {
        if (!id || !task) return

        try {
            const selectedContent = selectedModel === 'local' ? localResponse : gptResponse
            const selectedResponseMessage: Message = {
                role: 'assistant',
                content: selectedContent,
                timestamp: new Date().toISOString()
            }

            // Add the selected assistant message to the UI
            setTaskMessages(prev => [...prev, selectedResponseMessage as TaskMessage])

            // Show a processing state immediately
            setTask(prev => prev ? { ...prev, current_state: 'processing' } : null)

            console.log('🔄 Sending model selection to backend...')

            // Use the selectModel function from the hook
            await selectModel(selectedModel, id)

            console.log('✅ Model selection sent to backend - UI will update when tool result is received')

        } catch (error) {
            console.error('Error selecting model:', error)
            // On error, reload to get the correct state
            loadTask()
        }
    }

    // Effect to handle seamless tool result updates
    useEffect(() => {
        if (toolResultMessage && task) {
            const newToolMessage: TaskMessage = {
                ...toolResultMessage,
                timestamp: new Date(toolResultMessage.timestamp || Date.now()).toISOString()
            }

            // Add the tool result to the message list
            setTaskMessages(prev => [...prev, newToolMessage])
            setTask(prev => prev ? { ...prev, current_state: 'idle' } : null)


            const isAfterToolResponse = (
                newToolMessage.role === 'user' &&
                newToolMessage.content.includes('<tool_response>') &&
                taskMessages[taskMessages.length - 1]?.role === 'assistant'
            )

            if (isAfterToolResponse) {
                console.log('🚀 Auto-triggering AI response after seamless tool update')
                sendMessage(
                    newToolMessage.content,
                    task.model,
                    [...taskMessages, newToolMessage] as Message[],
                    id,
                    true
                )
            }
        }
    }, [toolResultMessage, sendMessage, id, task, taskMessages])

    // Auto-scroll to bottom when messages or responses change
    useEffect(() => {
        setTimeout(scrollToBottom, 100)
    }, [taskMessages, pendingDualResponses, currentResponse, localResponse, gptResponse])

    // REMOVED: Auto-reload during streaming - useWebSocket handles this now
    // The database approach was causing conflicts with the streaming approach

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center p-8">
                <motion.div
                    className="text-center"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                >
                    <div className="text-xl font-mono glow">
                        Loading task...
                    </div>
                </motion.div>
            </div>
        )
    }

    if (error || !task) {
        return (
            <div className="min-h-screen flex items-center justify-center p-8">
                <motion.div
                    className="text-center"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                >
                    <div className="text-xl font-mono glow text-red-400">
                        {error || 'Task not found'}
                    </div>
                    <button
                        onClick={() => navigate('/')}
                        className="mt-4 text-sm font-mono glow border rounded px-4 py-2"
                        style={{
                            borderColor: 'rgba(97, 253, 252, 0.3)',
                            color: '#61FDFC'
                        }}
                    >
                        ← Back to Home
                    </button>
                </motion.div>
            </div>
        )
    }

    return (
        <div className="min-h-screen relative">
            {/* Header with navigation */}
            <motion.div
                className="absolute z-10"
                style={{ top: '24px', left: '24px' }}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
            >
                <button
                    onClick={() => navigate('/')}
                    className="text-xs px-3 py-2 border rounded font-mono backdrop-blur"
                    style={{
                        backgroundColor: 'rgba(0, 0, 0, 0.4)',
                        borderColor: 'rgba(107, 114, 128, 0.3)',
                        color: '#61FDFC'
                    }}
                >
                    ← Back to Home
                </button>
            </motion.div>

            {/* Task Content */}
            <div className="p-8 pt-20">
                <div className="max-w-4xl mx-auto w-full">
                    {/* Task Title */}
                    <motion.div
                        className="text-center"
                        style={{ marginBottom: '50px' }}
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <h1 className="text-xl font-mono glow">{task.title}</h1>
                        <p className="text-sm font-mono opacity-60 mt-2">
                            State: {task.current_state} | Created: {new Date(task.created_at).toLocaleString()}
                        </p>
                    </motion.div>

                    {/* Task Messages History */}
                    {taskMessages.length > 0 && (
                        <motion.div
                            style={{ marginBottom: '40px' }}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: 0.2 }}
                        >
                            <div>
                                {taskMessages.map((message, index) => (
                                    <div
                                        key={index}
                                        className="text-base font-mono glow whitespace-pre-wrap leading-relaxed"
                                        style={{ marginBottom: '40px' }}
                                    >
                                        <div className={`mb-2 ${message.role === 'user' ? 'text-cyan-300' : 'text-green-300'}`}>
                                            {message.role === 'user' ? 'User' : 'Assistant'}
                                        </div>
                                        <div className="border-l ml-2" style={{
                                            paddingLeft: '20px',
                                            fontWeight: '300',
                                            color: '#E5E7EB',
                                            fontSize: '0.9rem',
                                            borderLeftColor: '#61FDFC',
                                            borderLeftWidth: '2px'
                                        }}>
                                            {parseContent(message.content)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    )}

                    {/* Streaming Dual Responses (from useWebSocket) */}
                    {isDualResponse && (
                        <motion.div
                            className="text-base font-mono glow"
                            style={{ marginBottom: '40px' }}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            <div className="text-green-300 mb-4">
                                {localFinished && gptFinished ? 'Choose the better response:' : 'Generating dual responses...'}
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* Local Model Response */}
                                <div
                                    className={`border rounded-lg p-4 cursor-pointer transition-all duration-300 ${localFinished ? 'border-green-400 hover:border-green-300' : 'border-gray-600'
                                        }`}
                                    onClick={() => {
                                        if (localFinished && gptFinished) {
                                            handleModelSelection('local')
                                        }
                                    }}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm font-medium text-green-400">
                                            Local Model
                                        </span>
                                        {localFinished && gptFinished && (
                                            <span className="text-xs text-green-300">✅ Click to Use</span>
                                        )}
                                        {!localFinished && (
                                            <span className="text-xs text-gray-400">🔄 Generating...</span>
                                        )}
                                    </div>
                                    <div className="text-sm text-gray-100 whitespace-pre-wrap">
                                        {parseContent(localResponse || 'Generating...')}
                                        {!localFinished && (
                                            <span className="animate-pulse ml-2">▋</span>
                                        )}
                                    </div>
                                </div>

                                {/* GPT Model Response */}
                                <div
                                    className={`border rounded-lg p-4 cursor-pointer transition-all duration-300 ${gptFinished ? 'border-blue-400 hover:border-blue-300' : 'border-gray-600'
                                        }`}
                                    onClick={() => {
                                        if (localFinished && gptFinished) {
                                            handleModelSelection('gpt')
                                        }
                                    }}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm font-medium text-blue-400">
                                            GPT Model
                                        </span>
                                        {gptFinished && localFinished && (
                                            <span className="text-xs text-blue-300">✅ Click to Use</span>
                                        )}
                                        {!gptFinished && (
                                            <span className="text-xs text-gray-400">🔄 Generating...</span>
                                        )}
                                    </div>
                                    <div className="text-sm text-gray-100 whitespace-pre-wrap">
                                        {parseContent(gptResponse || 'Generating...')}
                                        {!gptFinished && (
                                            <span className="animate-pulse ml-2">▋</span>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {/* Pending Dual Responses (from database) */}
                    {!isDualResponse && pendingDualResponses && task.current_state === 'awaiting_dual_selection' && (
                        <motion.div
                            className="text-base font-mono glow"
                            style={{ marginBottom: '40px' }}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            <div className="text-green-300 mb-4">
                                Choose the better response:
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* Local Model Response */}
                                <div
                                    className={`border rounded-lg p-4 cursor-pointer transition-all duration-300 ${pendingDualResponses.local_finished ? 'border-green-400 hover:border-green-300' : 'border-gray-600'
                                        }`}
                                    onClick={() => {
                                        if (pendingDualResponses.local_finished) {
                                            handleModelSelection('local')
                                        }
                                    }}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm font-medium text-green-400">
                                            {pendingDualResponses.local_model || 'Local Model'}
                                        </span>
                                        {pendingDualResponses.local_finished && (
                                            <span className="text-xs text-green-300">✅ Click to Use</span>
                                        )}
                                    </div>
                                    <div className="text-sm text-gray-100 whitespace-pre-wrap">
                                        {parseContent(pendingDualResponses.local_response || 'Generating...')}
                                    </div>
                                </div>

                                {/* GPT Model Response */}
                                <div
                                    className={`border rounded-lg p-4 cursor-pointer transition-all duration-300 ${pendingDualResponses.gpt_finished ? 'border-blue-400 hover:border-blue-300' : 'border-gray-600'
                                        }`}
                                    onClick={() => {
                                        if (pendingDualResponses.gpt_finished) {
                                            handleModelSelection('gpt')
                                        }
                                    }}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm font-medium text-blue-400">
                                            {pendingDualResponses.gpt_model || 'GPT Model'}
                                        </span>
                                        {pendingDualResponses.gpt_finished && (
                                            <span className="text-xs text-blue-300">✅ Click to Use</span>
                                        )}
                                    </div>
                                    <div className="text-sm text-gray-100 whitespace-pre-wrap">
                                        {parseContent(pendingDualResponses.gpt_response || 'Generating...')}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {/* Current Response (for single model streaming) */}
                    {currentResponse && task.current_state === 'streaming_single' && (
                        <motion.div
                            className="text-base font-mono glow"
                            style={{ marginBottom: '40px' }}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            <div className="text-green-300 mb-2">Assistant</div>
                            <div className="border-l ml-2" style={{
                                paddingLeft: '20px',
                                fontWeight: '300',
                                color: '#E5E7EB',
                                fontSize: '0.9rem',
                                borderLeftColor: '#61FDFC',
                                borderLeftWidth: '2px'
                            }}>
                                {parseContent(currentResponse)}
                                {isStreaming && (
                                    <span className="animate-pulse ml-2">▋</span>
                                )}
                            </div>
                        </motion.div>
                    )}

                    {/* Processing State */}
                    {task.current_state === 'processing' && !isDualResponse && (
                        <motion.div
                            className="text-center text-gray-400 font-mono"
                            style={{ marginBottom: '40px' }}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            <div className="animate-pulse">⚙️ Processing your selection...</div>
                            <div className="text-xs mt-2 opacity-60">
                                Executing tools and preparing next response
                            </div>
                        </motion.div>
                    )}

                    {/* REMOVED: Database-based streaming status - now using useWebSocket exclusively */}

                    {/* Auto-scroll anchor */}
                    <div ref={bottomRef} />
                </div>
            </div>

            {/* Sticky Input Box */}
            <div className="sticky bottom-0 left-0 right-0 bg-black/50 backdrop-blur-sm p-4">
                <form
                    onSubmit={handleSubmit}
                    className="mx-auto"
                    style={{ width: '800px', maxWidth: '90vw' }}
                >
                    <div className="relative">
                        <textarea
                            value={userInput}
                            onChange={(e) => setUserInput(e.target.value)}
                            placeholder={isDualResponse ? "Not quite right? Add a comment to clarify..." : "Send a follow-up message..."}
                            className="form-textarea min-h-[80px] w-full glow resize-none font-mono text-lg"
                            onKeyPress={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    (e.target as HTMLTextAreaElement).closest('form')?.requestSubmit();
                                }
                            }}
                        />
                        <div className="absolute bottom-3 right-3 text-xs text-gray-600">
                            Press Enter to send
                        </div>
                    </div>
                </form>
            </div>
        </div>
    )
}

export default TaskPage

