import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { useParams, useNavigate } from 'react-router-dom'
import TypingAnimation from './TypingAnimation'
import MCPServerDropdown from './MCPServerDropdown'
import useWebSocket, { ConnectionStatus } from '../hooks/useWebSocket'

interface Task {
    id: string
    title: string
    initial_prompt: string
    model: string
    mcp_servers: string[]
    created_at: string
    updated_at: string
    status: string
}

interface TaskMessage {
    role: string
    content: string
    timestamp: string
}

// Helper function to parse and render content with special tags
function parseContent(content: string): React.JSX.Element {
    const parts: React.JSX.Element[] = []
    let currentIndex = 0

    // Find all <think> tags
    const thinkRegex = /<think>(.*?)<\/think>/gs
    const toolCallRegex = /<tool_call>(.*?)<\/tool_call>/gs

    // Combine both patterns and find all matches with their positions
    const allMatches: Array<{ match: RegExpExecArray; type: 'think' | 'tool_call' }> = []

    let match
    while ((match = thinkRegex.exec(content)) !== null) {
        allMatches.push({ match, type: 'think' })
    }

    while ((match = toolCallRegex.exec(content)) !== null) {
        allMatches.push({ match, type: 'tool_call' })
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
                    fontSize: '0.85em',
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
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        border: '1px solid rgba(59, 130, 246, 0.3)',
                        borderRadius: '8px',
                        padding: '12px',
                        marginTop: '8px',
                        marginBottom: '8px',
                        fontSize: '0.9em'
                    }}>
                        <div style={{ color: '#3B82F6', fontWeight: 'bold', marginBottom: '4px' }}>
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
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        border: '1px solid rgba(59, 130, 246, 0.3)',
                        borderRadius: '8px',
                        padding: '12px',
                        marginTop: '8px',
                        marginBottom: '8px',
                        fontSize: '0.9em'
                    }}>
                        <div style={{ color: '#3B82F6', fontWeight: 'bold', marginBottom: '4px' }}>
                            🔧 Tool Call
                        </div>
                        <div style={{ color: '#E5E7EB', fontSize: '0.85em' }}>
                            {match[1]}
                        </div>
                    </div>
                )
            }
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
    const [prompt, setPrompt] = useState('')
    const [showMCPDropdown, setShowMCPDropdown] = useState(false)
    const [mcpQuery, setMcpQuery] = useState('')
    const [cursorPosition, setCursorPosition] = useState(0)
    const [isResponseComplete, setIsResponseComplete] = useState(false)
    const [task, setTask] = useState<Task | null>(null)
    const [taskMessages, setTaskMessages] = useState<TaskMessage[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const continueTextareaRef = useRef<HTMLTextAreaElement>(null)

    const {
        connectionStatus,
        currentResponse,
        isStreaming,
        mcpServers,
        loadingMcpServers,
        mcpServerActions,
        loadingMcpServerActions,
        connect,
        sendMessage,
        getMcpServers,
        getMcpServerActions,
        getTask,
        clearCurrentResponse,
        sendLearningFeedback
    } = useWebSocket({
        url: 'ws://localhost:8000',
        autoConnect: true,
        reconnectAttempts: 3,
        reconnectInterval: 3000
    })

    // Auto-connect when component mounts
    useEffect(() => {
        connect()
    }, [connect])

    // Load task data when component mounts or ID changes
    useEffect(() => {
        if (!id || connectionStatus !== ConnectionStatus.CONNECTED) {
            return
        }

        const loadTask = async (retryCount = 0) => {
            try {
                setLoading(true)
                setError(null)

                // Use the getTask method from the hook
                const taskData = await getTask(id)

                if (!taskData) {
                    // Retry if it's a connection issue and we haven't exceeded retry limit
                    if (retryCount < 3) {
                        setTimeout(() => loadTask(retryCount + 1), 500)
                        return
                    }

                    setError('Failed to load task')
                    setLoading(false)
                    return
                }

                if (taskData.error) {
                    setError(taskData.error)
                    setLoading(false)
                    return
                }

                setTask(taskData.task)
                setTaskMessages(taskData.messages || [])
                setLoading(false)

            } catch (error) {
                console.error('Error loading task:', error)

                // Retry if it's a connection issue and we haven't exceeded retry limit  
                if (retryCount < 3 && error instanceof Error && error.message.includes('CONNECTING')) {
                    setTimeout(() => loadTask(retryCount + 1), 500)
                    return
                }

                setError('Failed to load task')
                setLoading(false)
            }
        }

        loadTask()
    }, [id, connectionStatus, getTask])

    // Auto-trigger response for new tasks (separate effect to avoid infinite loops)
    useEffect(() => {
        if (!task || !taskMessages || taskMessages.length === 0 || isStreaming || currentResponse) {
            return
        }

        // Check if this task needs an initial AI response
        const hasOnlyUserMessages = taskMessages.length > 0 &&
            taskMessages.every((msg: TaskMessage) => msg.role === 'user')

        if (hasOnlyUserMessages && connectionStatus === ConnectionStatus.CONNECTED) {
            // Create message history from database messages 
            const messageHistory = taskMessages.map((msg: TaskMessage) => ({
                role: msg.role as 'user' | 'assistant' | 'system',
                content: msg.content,
                timestamp: Date.now()
            }))

            // Get the last user message content
            const lastUserMessage = taskMessages[taskMessages.length - 1]

            // Send auto-response with existing message history
            // The auto_response flag tells server not to save user message again
            sendMessage(lastUserMessage.content, 'qwen2.5-vl', messageHistory, id, true)
        }
    }, [task, taskMessages, isStreaming, currentResponse, connectionStatus, sendMessage, id])

    // Reset response complete state when streaming starts, set complete when streaming ends
    useEffect(() => {
        if (isStreaming) {
            setIsResponseComplete(false)
        } else if (currentResponse && !isStreaming) {
            // Small delay to ensure smooth transition
            setTimeout(() => setIsResponseComplete(true), 500)

            // Reload task messages after streaming completes to get the full updated conversation
            if (id && connectionStatus === ConnectionStatus.CONNECTED) {
                setTimeout(async () => {
                    try {
                        const taskData = await getTask(id)
                        if (taskData && !taskData.error) {
                            setTaskMessages(taskData.messages || [])
                            // Clear the current response to prevent duplicate display
                            clearCurrentResponse()
                        }
                    } catch (error) {
                        console.error('Error reloading task messages:', error)
                    }
                }, 1000) // Wait a bit longer to ensure server has saved the assistant response
            }
        }
    }, [isStreaming, currentResponse, id, connectionStatus, getTask, clearCurrentResponse])

    // Detect @ symbol and manage MCP dropdown
    const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const value = e.target.value
        const cursor = e.target.selectionStart

        setPrompt(value)
        setCursorPosition(cursor)

        // Look for @ symbol
        const atIndex = value.lastIndexOf('@', cursor - 1)

        if (atIndex !== -1) {
            // Check if @ is at start of word (preceded by space or start of string)
            const charBefore = atIndex === 0 ? ' ' : value[atIndex - 1]
            if (charBefore === ' ' || charBefore === '\n' || atIndex === 0) {
                const afterAt = value.substring(atIndex + 1, cursor)
                // Only show if no spaces after @
                if (!afterAt.includes(' ') && !afterAt.includes('\n')) {
                    setMcpQuery(afterAt)
                    setShowMCPDropdown(true)
                    getMcpServers(afterAt)
                    return
                }
            }
        }

        // Hide dropdown if no valid @ context
        setShowMCPDropdown(false)
    }

    // Handle MCP server selection
    const handleMCPServerSelect = (server: any) => {
        const atIndex = prompt.lastIndexOf('@', cursorPosition - 1)
        if (atIndex !== -1) {
            const beforeAt = prompt.substring(0, atIndex)
            const afterCursor = prompt.substring(cursorPosition)
            const newPrompt = `${beforeAt}@${server.id} ${afterCursor}`
            setPrompt(newPrompt)

            // Focus textarea and position cursor after the inserted server
            setTimeout(() => {
                if (continueTextareaRef.current) {
                    const newCursorPos = atIndex + server.id.length + 2
                    continueTextareaRef.current.focus()
                    continueTextareaRef.current.setSelectionRange(newCursorPos, newCursorPos)
                }
            }, 0)
        }

        setShowMCPDropdown(false)
    }

    // Close MCP dropdown
    const closeMCPDropdown = () => {
        setShowMCPDropdown(false)
    }

    // Update MCP search when servers list changes
    useEffect(() => {
        if (showMCPDropdown && connectionStatus === ConnectionStatus.CONNECTED) {
            getMcpServers(mcpQuery)
        }
    }, [mcpQuery, showMCPDropdown, connectionStatus, getMcpServers])

    // Handle continuation of conversation
    const handleContinue = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault()
        if (!prompt.trim() || connectionStatus !== ConnectionStatus.CONNECTED || isStreaming || !id) {
            return
        }

        const newUserMessage = {
            role: 'user',
            content: prompt.trim(),
            timestamp: new Date().toISOString()
        }

        // Immediately add user message to display
        setTaskMessages(prev => [...prev, newUserMessage])

        // Build messages array from task history + new message
        const messages = [
            ...taskMessages.map(msg => ({
                role: msg.role as 'user' | 'assistant' | 'system',
                content: msg.content,
                timestamp: Date.now()
            })),
            {
                role: 'user' as const,
                content: prompt.trim(),
                timestamp: Date.now()
            }
        ]

        // Send message with task_id for persistence
        sendMessage(prompt.trim(), 'qwen2.5-vl', messages, id)
        setPrompt('')
        setIsResponseComplete(false) // Reset for new response
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            const form = e.currentTarget.closest('form')
            if (form) {
                const event = new Event('submit', { bubbles: true, cancelable: true })
                form.dispatchEvent(event)
            }
        }
    }

    // State to track which button is being hovered
    const [hoveredButton, setHoveredButton] = useState<string | null>(null)

    // State for inline comments
    const [activeCommentBox, setActiveCommentBox] = useState<string | null>(null)
    const [commentText, setCommentText] = useState('')
    const [submittedComments, setSubmittedComments] = useState<Record<string, string>>({})

    // State to track clicked buttons (to keep them highlighted)
    const [clickedButtons, setClickedButtons] = useState<Set<string>>(new Set())

    // State for MCP server actions selection
    const [showMcpActionsBox, setShowMcpActionsBox] = useState<string | null>(null)
    const [selectedAction, setSelectedAction] = useState<{ serverId: string; actionName: string; description: string } | null>(null)

    // Handle learning feedback for assistant messages
    const handleLearningFeedback = async (type: string, message: TaskMessage, index: number) => {
        if (!id) return

        try {
            let userComment: string | undefined
            const buttonId = getButtonId(index, type)

            // Mark button as clicked
            setClickedButtons(prev => new Set(prev).add(buttonId))

            if (type === 'deny' || type === 'comment') {
                // Show inline comment box for these types
                setActiveCommentBox(buttonId)
                setCommentText('')
                return // Don't send feedback yet, wait for comment submission
            }

            if (type === 'correct') {
                // Show MCP server actions selection
                setShowMcpActionsBox(buttonId)

                // Get MCP server actions for this task
                if (task?.mcp_servers && task.mcp_servers.length > 0) {
                    getMcpServerActions(task.mcp_servers)
                }
                return // Don't send feedback yet, wait for action selection
            }

            await sendLearningFeedback(type, message, id, userComment)

            // Show user confirmation
            console.log(`Learning feedback sent: ${type} for message at index ${index}`)

        } catch (error) {
            console.error('Failed to send learning feedback:', error)
        }
    }

    // Handle current response learning feedback
    const handleCurrentResponseLearningFeedback = async (type: string) => {
        if (!id || !currentResponse) return

        try {
            let userComment: string | undefined
            const buttonId = getCurrentButtonId(type)

            // Mark button as clicked
            setClickedButtons(prev => new Set(prev).add(buttonId))

            if (type === 'deny' || type === 'comment') {
                // Show inline comment box for these types
                setActiveCommentBox(buttonId)
                setCommentText('')
                return // Don't send feedback yet, wait for comment submission
            }

            if (type === 'correct') {
                // Show MCP server actions selection
                setShowMcpActionsBox(buttonId)

                // Get MCP server actions for this task
                if (task?.mcp_servers && task.mcp_servers.length > 0) {
                    getMcpServerActions(task.mcp_servers)
                }
                return // Don't send feedback yet, wait for action selection
            }

            const message = { role: 'assistant', content: currentResponse, timestamp: new Date().toISOString() }
            await sendLearningFeedback(type, message, id, userComment)

            // Show user confirmation
            console.log(`Learning feedback sent: ${type} for current response`)

        } catch (error) {
            console.error('Failed to send learning feedback:', error)
        }
    }

    // Handle MCP action selection
    const handleMcpActionSelect = async (buttonId: string, messageIndex: number, message?: TaskMessage) => {
        if (!selectedAction) return

        try {
            const actionDescription = `Use ${selectedAction.serverId} action "${selectedAction.actionName}": ${selectedAction.description}`

            const isCurrentResponse = buttonId.startsWith('current-')
            let messageToSend: TaskMessage
            if (isCurrentResponse) {
                messageToSend = { role: 'assistant', content: currentResponse, timestamp: new Date().toISOString() }
            } else {
                messageToSend = message!
            }

            await sendLearningFeedback('correct', messageToSend, id!, actionDescription)

            // Clear MCP actions box
            setShowMcpActionsBox(null)
            setSelectedAction(null)

            console.log(`MCP action selected for correction:`, actionDescription)

        } catch (error) {
            console.error('Failed to submit MCP action:', error)
        }
    }

    // Cancel MCP action selection
    const handleMcpActionCancel = () => {
        setShowMcpActionsBox(null)
        setSelectedAction(null)
    }

    // Handle comment submission
    const handleCommentSubmit = async (buttonId: string, messageIndex: number, message?: TaskMessage) => {
        if (!commentText.trim()) return

        try {
            const isCurrentResponse = buttonId.startsWith('current-')
            const feedbackType = isCurrentResponse
                ? buttonId.replace('current-', '')
                : buttonId.split('-').pop()

            let messageToSend: TaskMessage
            if (isCurrentResponse) {
                messageToSend = { role: 'assistant', content: currentResponse, timestamp: new Date().toISOString() }
            } else {
                messageToSend = message!
            }

            await sendLearningFeedback(feedbackType!, messageToSend, id!, commentText)

            // Store submitted comment for display
            setSubmittedComments(prev => ({
                ...prev,
                [buttonId]: commentText
            }))

            // Clear comment box
            setActiveCommentBox(null)
            setCommentText('')

            console.log(`Comment submitted for ${feedbackType}:`, commentText)

        } catch (error) {
            console.error('Failed to submit comment:', error)
        }
    }

    // Cancel comment
    const handleCommentCancel = () => {
        setActiveCommentBox(null)
        setCommentText('')
    }

    // Generate unique button ID for hover tracking
    const getButtonId = (messageIndex: number, buttonType: string) => `${messageIndex}-${buttonType}`
    const getCurrentButtonId = (buttonType: string) => `current-${buttonType}`

    const getStatusColor = () => {
        switch (connectionStatus) {
            case ConnectionStatus.CONNECTED:
                return 'status-online'
            case ConnectionStatus.CONNECTING:
                return 'status-connecting'
            case ConnectionStatus.ERROR:
            case ConnectionStatus.DISCONNECTED:
                return 'status-offline'
            default:
                return 'status-offline'
        }
    }

    const getStatusText = () => {
        switch (connectionStatus) {
            case ConnectionStatus.CONNECTED:
                return 'CONNECTED'
            case ConnectionStatus.CONNECTING:
                return 'CONNECTING'
            case ConnectionStatus.ERROR:
                return 'ERROR'
            case ConnectionStatus.DISCONNECTED:
                return 'DISCONNECTED'
            default:
                return 'UNKNOWN'
        }
    }

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
            {/* Connection Status - Top Right */}
            <motion.div
                className="absolute z-10"
                style={{ top: '24px', right: '24px' }}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
            >
                <div
                    className="backdrop-blur border flex items-center"
                    style={{
                        backgroundColor: 'rgba(0, 0, 0, 0.4)',
                        borderColor: 'rgba(107, 114, 128, 0.3)',
                        padding: '8px 16px',
                        gap: '12px',
                        borderRadius: '16px'
                    }}
                >
                    <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full glow`} style={{ backgroundColor: '#61FDFC' }}></div>
                        <span className={`text-xs font-mono ${getStatusColor()}`}>
                            {getStatusText()}
                        </span>
                    </div>
                    {connectionStatus !== ConnectionStatus.CONNECTED && connectionStatus !== ConnectionStatus.CONNECTING && (
                        <button
                            onClick={connect}
                            className="text-xs px-2 py-1 border rounded font-mono"
                            style={{
                                backgroundColor: 'rgba(97, 253, 252, 0.2)',
                                borderColor: 'rgba(97, 253, 252, 0.3)',
                                color: '#61FDFC'
                            }}
                        >
                            CONNECT
                        </button>
                    )}
                </div>
            </motion.div>

            {/* Back Button - Top Left */}
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
            {connectionStatus === ConnectionStatus.CONNECTED ? (
                <div className="min-h-screen p-8 pt-20 flex items-center justify-center">
                    <div className="max-w-4xl mx-auto w-full response-container">

                        {/* Task Title */}
                        <motion.div
                            className="text-center"
                            style={{ marginBottom: '100px' }}
                            initial={{ opacity: 0, y: -20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            <h1 className="text-xl font-mono glow">{task.title}</h1>
                            <p className="text-sm font-mono opacity-60 mt-2">
                                Created: {new Date(task.created_at).toLocaleString()}
                            </p>
                        </motion.div>

                        {/* Task Messages History */}
                        {taskMessages.length > 0 && (
                            <motion.div
                                style={{ marginBottom: '80px' }}
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
                                            {message.role === 'user' ? (
                                                <div className="text-cyan-300 mb-2">User</div>
                                            ) : (
                                                <div className="text-green-300 mb-2">Assistant</div>
                                            )}
                                            <div className="border-l border-gray-600 ml-2" style={{ paddingLeft: '20px' }}>
                                                {parseContent(message.content)}
                                            </div>

                                            {/* Learning buttons for assistant messages */}
                                            {message.role === 'assistant' && (
                                                <>
                                                    <div className="flex gap-4 mb-2" style={{ marginTop: '32px', marginLeft: '24px' }}>
                                                        <button
                                                            onClick={() => handleLearningFeedback('approve', message, index)}
                                                            onMouseEnter={() => setHoveredButton(getButtonId(index, 'approve'))}
                                                            onMouseLeave={() => setHoveredButton(null)}
                                                            className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                            style={{
                                                                backgroundColor: 'rgba(0, 0, 0, 0.2)',
                                                                borderColor: (hoveredButton === getButtonId(index, 'approve') || clickedButtons.has(getButtonId(index, 'approve'))) ? '#61FDFC' : 'rgba(107, 114, 128, 0.3)',
                                                                color: '#9CA3AF',
                                                                borderRadius: '12px',
                                                                padding: '16px 32px',
                                                                cursor: 'pointer'
                                                            }}
                                                        >
                                                            <span style={{ color: '#22c55e' }}>✓</span> Approve
                                                        </button>
                                                        <button
                                                            onClick={() => handleLearningFeedback('deny', message, index)}
                                                            onMouseEnter={() => setHoveredButton(getButtonId(index, 'deny'))}
                                                            onMouseLeave={() => setHoveredButton(null)}
                                                            className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                            style={{
                                                                backgroundColor: 'rgba(0, 0, 0, 0.2)',
                                                                borderColor: (hoveredButton === getButtonId(index, 'deny') || clickedButtons.has(getButtonId(index, 'deny'))) ? '#61FDFC' : 'rgba(107, 114, 128, 0.3)',
                                                                color: '#9CA3AF',
                                                                borderRadius: '12px',
                                                                padding: '16px 32px',
                                                                cursor: 'pointer'
                                                            }}
                                                        >
                                                            <span style={{ color: '#ef4444' }}>✗</span> Deny
                                                        </button>
                                                        <button
                                                            onClick={() => handleLearningFeedback('correct', message, index)}
                                                            onMouseEnter={() => setHoveredButton(getButtonId(index, 'correct'))}
                                                            onMouseLeave={() => setHoveredButton(null)}
                                                            className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                            style={{
                                                                backgroundColor: 'rgba(0, 0, 0, 0.2)',
                                                                borderColor: (hoveredButton === getButtonId(index, 'correct') || clickedButtons.has(getButtonId(index, 'correct'))) ? '#61FDFC' : 'rgba(107, 114, 128, 0.3)',
                                                                color: '#9CA3AF',
                                                                borderRadius: '12px',
                                                                padding: '16px 32px',
                                                                cursor: 'pointer'
                                                            }}
                                                        >
                                                            <span style={{ color: '#3b82f6' }}>✎</span> Correct
                                                        </button>
                                                        <button
                                                            onClick={() => handleLearningFeedback('comment', message, index)}
                                                            onMouseEnter={() => setHoveredButton(getButtonId(index, 'comment'))}
                                                            onMouseLeave={() => setHoveredButton(null)}
                                                            className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                            style={{
                                                                backgroundColor: 'rgba(0, 0, 0, 0.2)',
                                                                borderColor: (hoveredButton === getButtonId(index, 'comment') || clickedButtons.has(getButtonId(index, 'comment'))) ? '#61FDFC' : 'rgba(107, 114, 128, 0.3)',
                                                                color: '#9CA3AF',
                                                                borderRadius: '12px',
                                                                padding: '16px 32px',
                                                                cursor: 'pointer'
                                                            }}
                                                        >
                                                            <span style={{ color: '#a855f7' }}>💬</span> Comment
                                                        </button>
                                                    </div>

                                                    {/* Inline comment box for historical messages */}
                                                    {(['deny', 'comment'].some(type => activeCommentBox === getButtonId(index, type))) && (
                                                        <div style={{ marginTop: '16px', marginLeft: '24px' }}>
                                                            <textarea
                                                                value={commentText}
                                                                onChange={(e) => setCommentText(e.target.value)}
                                                                placeholder={`Add your ${activeCommentBox?.includes('deny') ? 'feedback on why this was wrong' : 'comment'}...`}
                                                                className="form-textarea min-h-[80px] w-full glow resize-none font-mono text-sm"
                                                                style={{ maxWidth: '500px' }}
                                                            />
                                                            <div className="flex gap-2 mt-2">
                                                                <button
                                                                    onClick={() => handleCommentSubmit(activeCommentBox!, index, message)}
                                                                    className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                                    style={{
                                                                        backgroundColor: 'rgba(34, 197, 94, 0.2)',
                                                                        borderColor: 'rgba(34, 197, 94, 0.4)',
                                                                        color: '#22c55e',
                                                                        borderRadius: '8px',
                                                                        padding: '8px 16px',
                                                                        cursor: 'pointer'
                                                                    }}
                                                                >
                                                                    Submit
                                                                </button>
                                                                <button
                                                                    onClick={handleCommentCancel}
                                                                    className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                                    style={{
                                                                        backgroundColor: 'rgba(107, 114, 128, 0.2)',
                                                                        borderColor: 'rgba(107, 114, 128, 0.3)',
                                                                        color: '#9CA3AF',
                                                                        borderRadius: '8px',
                                                                        padding: '8px 16px',
                                                                        cursor: 'pointer'
                                                                    }}
                                                                >
                                                                    Cancel
                                                                </button>
                                                            </div>
                                                        </div>
                                                    )}

                                                    {/* MCP Server Actions selection box for historical messages */}
                                                    {showMcpActionsBox === getButtonId(index, 'correct') && (
                                                        <div style={{ marginTop: '16px', marginLeft: '24px' }}>
                                                            <div className="font-mono text-sm glow mb-2" style={{ color: '#61FDFC' }}>
                                                                Select the correct action from available MCP servers:
                                                            </div>

                                                            {loadingMcpServerActions ? (
                                                                <div style={{ color: '#9CA3AF', fontSize: '12px' }}>Loading actions...</div>
                                                            ) : (
                                                                <div style={{
                                                                    maxHeight: '200px',
                                                                    overflowY: 'auto',
                                                                    backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                                                    border: '1px solid rgba(107, 114, 128, 0.3)',
                                                                    borderRadius: '8px',
                                                                    padding: '8px',
                                                                    maxWidth: '500px'
                                                                }}>
                                                                    {task?.mcp_servers && task.mcp_servers.map(serverId => (
                                                                        <div key={serverId} style={{ marginBottom: '12px' }}>
                                                                            <div style={{ color: '#3B82F6', fontSize: '12px', marginBottom: '4px', fontWeight: 'bold' }}>
                                                                                {serverId}
                                                                            </div>
                                                                            {mcpServerActions[serverId]?.map(action => (
                                                                                <button
                                                                                    key={action.name}
                                                                                    onClick={() => setSelectedAction({ serverId, actionName: action.name, description: action.description })}
                                                                                    className="text-xs font-mono border transition-all duration-200 hover:scale-105 block w-full text-left mb-1"
                                                                                    style={{
                                                                                        backgroundColor: selectedAction?.serverId === serverId && selectedAction?.actionName === action.name
                                                                                            ? 'rgba(59, 130, 246, 0.2)' : 'rgba(0, 0, 0, 0.2)',
                                                                                        borderColor: selectedAction?.serverId === serverId && selectedAction?.actionName === action.name
                                                                                            ? '#3B82F6' : 'rgba(107, 114, 128, 0.3)',
                                                                                        color: '#E5E7EB',
                                                                                        borderRadius: '4px',
                                                                                        padding: '8px',
                                                                                        cursor: 'pointer'
                                                                                    }}
                                                                                >
                                                                                    <div style={{ fontWeight: 'bold' }}>{action.name}</div>
                                                                                    <div style={{ fontSize: '10px', opacity: 0.8 }}>{action.description}</div>
                                                                                </button>
                                                                            ))}
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            )}

                                                            <div className="flex gap-2 mt-2">
                                                                <button
                                                                    onClick={() => handleMcpActionSelect(showMcpActionsBox!, index, message)}
                                                                    disabled={!selectedAction}
                                                                    className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                                    style={{
                                                                        backgroundColor: selectedAction ? 'rgba(34, 197, 94, 0.2)' : 'rgba(107, 114, 128, 0.1)',
                                                                        borderColor: selectedAction ? 'rgba(34, 197, 94, 0.4)' : 'rgba(107, 114, 128, 0.3)',
                                                                        color: selectedAction ? '#22c55e' : '#6B7280',
                                                                        borderRadius: '8px',
                                                                        padding: '8px 16px',
                                                                        cursor: selectedAction ? 'pointer' : 'not-allowed'
                                                                    }}
                                                                >
                                                                    Use Selected Action
                                                                </button>
                                                                <button
                                                                    onClick={handleMcpActionCancel}
                                                                    className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                                    style={{
                                                                        backgroundColor: 'rgba(107, 114, 128, 0.2)',
                                                                        borderColor: 'rgba(107, 114, 128, 0.3)',
                                                                        color: '#9CA3AF',
                                                                        borderRadius: '8px',
                                                                        padding: '8px 16px',
                                                                        cursor: 'pointer'
                                                                    }}
                                                                >
                                                                    Cancel
                                                                </button>
                                                            </div>
                                                        </div>
                                                    )}

                                                    {/* Display submitted comments for historical messages */}
                                                    {(['deny', 'comment'].some(type => submittedComments[getButtonId(index, type)])) && (
                                                        <div style={{ marginTop: '16px', marginLeft: '24px' }}>
                                                            <div
                                                                className="font-mono text-sm glow"
                                                                style={{
                                                                    backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                                                    borderLeft: '3px solid #61FDFC',
                                                                    padding: '12px 16px',
                                                                    borderRadius: '8px',
                                                                    maxWidth: '500px'
                                                                }}
                                                            >
                                                                <div style={{ color: '#61FDFC', fontSize: '11px', marginBottom: '4px' }}>
                                                                    {submittedComments[getButtonId(index, 'deny')] ? 'FEEDBACK' : 'COMMENT'}
                                                                </div>
                                                                <div style={{ color: '#E5E7EB' }}>
                                                                    {submittedComments[getButtonId(index, 'deny')] || submittedComments[getButtonId(index, 'comment')]}
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}
                                                </>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </motion.div>
                        )}

                        {/* Current Streaming Response */}
                        {(currentResponse || isStreaming) && (
                            <div
                                className="text-base font-mono glow whitespace-pre-wrap leading-relaxed word-break-normal max-w-full overflow-wrap-anywhere"
                                style={{ marginBottom: '40px' }}
                            >
                                <div className="relative">
                                    <div className="text-green-300 mb-2">Assistant:</div>
                                    <div className="border-l border-gray-600 ml-2" style={{ paddingLeft: '20px' }}>
                                        <span className="max-w-full break-words">
                                            {parseContent(currentResponse)}
                                            {isStreaming && (
                                                <motion.span
                                                    className="typing-cursor inline"
                                                    animate={{ opacity: [1, 0] }}
                                                    transition={{ duration: 1, repeat: Infinity }}
                                                />
                                            )}
                                        </span>
                                    </div>

                                    {/* Learning buttons for current response (only when not streaming) */}
                                    {!isStreaming && currentResponse && (
                                        <div className="flex gap-4 mb-2" style={{ marginTop: '32px', marginLeft: '24px' }}>
                                            <button
                                                onClick={() => handleCurrentResponseLearningFeedback('approve')}
                                                onMouseEnter={() => setHoveredButton(getCurrentButtonId('approve'))}
                                                onMouseLeave={() => setHoveredButton(null)}
                                                className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                style={{
                                                    backgroundColor: 'rgba(0, 0, 0, 0.2)',
                                                    borderColor: (hoveredButton === getCurrentButtonId('approve') || clickedButtons.has(getCurrentButtonId('approve'))) ? '#61FDFC' : 'rgba(107, 114, 128, 0.3)',
                                                    color: '#9CA3AF',
                                                    borderRadius: '12px',
                                                    padding: '16px 32px',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                <span style={{ color: '#22c55e' }}>✓</span> Approve
                                            </button>
                                            <button
                                                onClick={() => handleCurrentResponseLearningFeedback('deny')}
                                                onMouseEnter={() => setHoveredButton(getCurrentButtonId('deny'))}
                                                onMouseLeave={() => setHoveredButton(null)}
                                                className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                style={{
                                                    backgroundColor: 'rgba(0, 0, 0, 0.2)',
                                                    borderColor: (hoveredButton === getCurrentButtonId('deny') || clickedButtons.has(getCurrentButtonId('deny'))) ? '#61FDFC' : 'rgba(107, 114, 128, 0.3)',
                                                    color: '#9CA3AF',
                                                    borderRadius: '12px',
                                                    padding: '16px 32px',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                <span style={{ color: '#ef4444' }}>✗</span> Deny
                                            </button>
                                            <button
                                                onClick={() => handleCurrentResponseLearningFeedback('correct')}
                                                onMouseEnter={() => setHoveredButton(getCurrentButtonId('correct'))}
                                                onMouseLeave={() => setHoveredButton(null)}
                                                className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                style={{
                                                    backgroundColor: 'rgba(0, 0, 0, 0.2)',
                                                    borderColor: (hoveredButton === getCurrentButtonId('correct') || clickedButtons.has(getCurrentButtonId('correct'))) ? '#61FDFC' : 'rgba(107, 114, 128, 0.3)',
                                                    color: '#9CA3AF',
                                                    borderRadius: '12px',
                                                    padding: '16px 32px',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                <span style={{ color: '#3b82f6' }}>✎</span> Correct
                                            </button>
                                            <button
                                                onClick={() => handleCurrentResponseLearningFeedback('comment')}
                                                onMouseEnter={() => setHoveredButton(getCurrentButtonId('comment'))}
                                                onMouseLeave={() => setHoveredButton(null)}
                                                className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                style={{
                                                    backgroundColor: 'rgba(0, 0, 0, 0.2)',
                                                    borderColor: (hoveredButton === getCurrentButtonId('comment') || clickedButtons.has(getCurrentButtonId('comment'))) ? '#61FDFC' : 'rgba(107, 114, 128, 0.3)',
                                                    color: '#9CA3AF',
                                                    borderRadius: '12px',
                                                    padding: '16px 32px',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                <span style={{ color: '#a855f7' }}>💬</span> Comment
                                            </button>
                                        </div>
                                    )}

                                    {/* Inline comment box for current response */}
                                    {(['deny', 'comment'].some(type => activeCommentBox === getCurrentButtonId(type))) && (
                                        <div style={{ marginTop: '16px', marginLeft: '24px' }}>
                                            <textarea
                                                value={commentText}
                                                onChange={(e) => setCommentText(e.target.value)}
                                                placeholder={`Add your ${activeCommentBox?.includes('deny') ? 'feedback on why this was wrong' : 'comment'}...`}
                                                className="form-textarea min-h-[80px] w-full glow resize-none font-mono text-sm"
                                                style={{ maxWidth: '500px' }}
                                            />
                                            <div className="flex gap-2 mt-2">
                                                <button
                                                    onClick={() => handleCommentSubmit(activeCommentBox!, -1)}
                                                    className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                    style={{
                                                        backgroundColor: 'rgba(34, 197, 94, 0.2)',
                                                        borderColor: 'rgba(34, 197, 94, 0.4)',
                                                        color: '#22c55e',
                                                        borderRadius: '8px',
                                                        padding: '8px 16px',
                                                        cursor: 'pointer'
                                                    }}
                                                >
                                                    Submit
                                                </button>
                                                <button
                                                    onClick={handleCommentCancel}
                                                    className="text-xs font-mono border transition-all duration-200 hover:scale-105"
                                                    style={{
                                                        backgroundColor: 'rgba(107, 114, 128, 0.2)',
                                                        borderColor: 'rgba(107, 114, 128, 0.3)',
                                                        color: '#9CA3AF',
                                                        borderRadius: '8px',
                                                        padding: '8px 16px',
                                                        cursor: 'pointer'
                                                    }}
                                                >
                                                    Cancel
                                                </button>
                                            </div>
                                        </div>
                                    )}

                                    {/* Display submitted comments for current response */}
                                    {(['deny', 'comment'].some(type => submittedComments[getCurrentButtonId(type)])) && (
                                        <div style={{ marginTop: '16px', marginLeft: '24px' }}>
                                            <div
                                                className="font-mono text-sm glow"
                                                style={{
                                                    backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                                    borderLeft: '3px solid #61FDFC',
                                                    padding: '12px 16px',
                                                    borderRadius: '8px',
                                                    maxWidth: '500px'
                                                }}
                                            >
                                                <div style={{ color: '#61FDFC', fontSize: '11px', marginBottom: '4px' }}>
                                                    {submittedComments[getCurrentButtonId('deny')] ? 'FEEDBACK' : 'COMMENT'}
                                                </div>
                                                <div style={{ color: '#E5E7EB' }}>
                                                    {submittedComments[getCurrentButtonId('deny')] || submittedComments[getCurrentButtonId('comment')]}
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Input Box - Always present with consistent spacing */}
                        <motion.div
                            className="max-w-2xl mx-auto"
                            style={{ marginTop: '120px' }}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            <form onSubmit={handleContinue} className="space-y-4">
                                <div className="relative">
                                    <textarea
                                        ref={continueTextareaRef}
                                        value={prompt}
                                        onChange={handleTextareaChange}
                                        onKeyPress={handleKeyPress}
                                        className="form-textarea min-h-[120px] w-full glow resize-none font-mono text-lg"
                                        placeholder="Continue the conversation... (type @ to mention MCP servers)"
                                        disabled={isStreaming}
                                    />
                                    <div className="absolute bottom-3 right-3 text-xs text-gray-600">
                                        Press Enter to send
                                    </div>

                                    {/* MCP Server Dropdown */}
                                    <MCPServerDropdown
                                        isVisible={showMCPDropdown}
                                        servers={mcpServers}
                                        loading={loadingMcpServers}
                                        onSelect={handleMCPServerSelect}
                                        onClose={closeMCPDropdown}
                                        query={mcpQuery}
                                    />
                                </div>
                            </form>
                        </motion.div>

                    </div>
                </div>
            ) : (
                // CONNECTION REQUIRED
                <div className="min-h-screen flex items-center justify-center p-8">
                    <motion.div
                        className="text-center"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5 }}
                    >
                        <div className="text-gray-600 glow font-mono">
                            {connectionStatus === ConnectionStatus.CONNECTING
                                ? "Establishing neural connection..."
                                : "Neural connection required"
                            }
                        </div>
                    </motion.div>
                </div>
            )}
        </div>
    )
}

export default TaskPage 