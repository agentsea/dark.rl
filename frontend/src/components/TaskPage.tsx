import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { useParams, useNavigate } from 'react-router-dom'
import TypingAnimation from './TypingAnimation'
import MCPServerDropdown from './MCPServerDropdown'
import CorrectionModal from './CorrectionModal'
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

// Enhanced version with consistent formatting for dual responses
function parseContentWithConsistentFormatting(content: string): React.JSX.Element {
    // Trim whitespace to ensure consistent formatting between models
    const trimmedContent = content.trim()

    const parts: React.JSX.Element[] = []
    let currentIndex = 0

    // Find all <think>, <tool_call>, and <tool_response> tags
    const thinkRegex = /<think>(.*?)<\/think>/gs
    const toolCallRegex = /<tool_call>(.*?)<\/tool_call>/gs
    const toolResponseRegex = /<tool_response>(.*?)<\/tool_response>/gs

    // Combine all patterns and find all matches with their positions
    const allMatches: Array<{ match: RegExpExecArray; type: 'think' | 'tool_call' | 'tool_response' }> = []

    let match
    while ((match = thinkRegex.exec(trimmedContent)) !== null) {
        allMatches.push({ match, type: 'think' })
    }

    while ((match = toolCallRegex.exec(trimmedContent)) !== null) {
        allMatches.push({ match, type: 'tool_call' })
    }

    while ((match = toolResponseRegex.exec(trimmedContent)) !== null) {
        allMatches.push({ match, type: 'tool_response' })
    }

    // Sort matches by position
    allMatches.sort((a, b) => a.match.index - b.match.index)

    // Process each match
    allMatches.forEach((matchObj, index) => {
        const { match, type } = matchObj

        // Add text before this match
        if (match.index > currentIndex) {
            const beforeText = trimmedContent.slice(currentIndex, match.index)
            // Normalize whitespace between sections - convert multiple newlines to single space
            const normalizedText = beforeText.replace(/\s+/g, ' ').trim()
            if (normalizedText) {
                parts.push(<span key={`text-${index}`}>{normalizedText}</span>)
            }
        }

        // Add the special tag content with consistent formatting
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
                    💭 {match[1].trim()}
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
                        <div style={{
                            color: '#61FDFC',
                            fontWeight: 'bold',
                            marginBottom: '4px'
                        }}>
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
                        <div style={{
                            color: '#61FDFC',
                            fontWeight: 'bold',
                            marginBottom: '4px'
                        }}>
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
    if (currentIndex < trimmedContent.length) {
        const remainingText = trimmedContent.slice(currentIndex)
        // Normalize whitespace in remaining text
        const normalizedRemaining = remainingText.replace(/\s+/g, ' ').trim()
        if (normalizedRemaining) {
            parts.push(<span key="remaining">{normalizedRemaining}</span>)
        }
    }

    // If no special tags found, return the original content
    if (parts.length === 0) {
        return <span>{trimmedContent}</span>
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
    const [isProcessingSelection, setIsProcessingSelection] = useState(false)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const continueTextareaRef = useRef<HTMLTextAreaElement>(null)
    const bottomRef = useRef<HTMLDivElement>(null)

    // Auto-scroll to bottom function
    const scrollToBottom = () => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    const {
        connectionStatus,
        currentResponse,
        isStreaming,
        mcpServers,
        loadingMcpServers,
        mcpServerActions,
        loadingMcpServerActions,
        // Dual response state
        isDualResponse,
        localResponse,
        gptResponse,
        localModel,
        gptModel,
        localFinished,
        gptFinished,
        connect,
        sendMessage,
        getMcpServers,
        getMcpServerActions,
        getTask,
        clearCurrentResponse,
        clearDualResponse,
        selectModel,
        sendLearningFeedback,
        sendCorrectionWithExecution
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

                // Load MCP server actions for this task
                if (taskData.task.mcp_servers && taskData.task.mcp_servers.length > 0) {
                    console.log('🛠️ Loading MCP server actions for:', taskData.task.mcp_servers)
                    getMcpServerActions(taskData.task.mcp_servers)
                }

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
    }, [id, connectionStatus, getTask, getMcpServerActions])

    // Auto-trigger response for new tasks (separate effect to avoid infinite loops)
    useEffect(() => {
        if (!task || !taskMessages || taskMessages.length === 0 || isStreaming || currentResponse || isProcessingSelection) {
            return
        }

        // Don't auto-trigger if we're in dual response mode or have dual responses
        if (isDualResponse || localResponse || gptResponse) {
            return
        }

        // Check if this task needs an initial AI response
        const hasOnlyUserMessages = taskMessages.length > 0 &&
            taskMessages.every((msg: TaskMessage) => msg.role === 'user')

        if (hasOnlyUserMessages) {
            console.log('🚀 Auto-triggering response for new task')

            // Get the last user message content
            const lastUserMessage = taskMessages[taskMessages.length - 1]

            // Convert task messages to the format expected by sendMessage
            const messageHistory = taskMessages.map((msg: TaskMessage) => ({
                role: msg.role as 'user' | 'assistant' | 'system',
                content: msg.content
            }))

            sendMessage(
                lastUserMessage.content,  // content as string
                'qwen2.5-vl',            // model
                messageHistory,          // customMessages
                id,                      // taskId
                true                     // isAutoResponse
            )
        }
    }, [task, taskMessages, isStreaming, currentResponse, isDualResponse, localResponse, gptResponse, isProcessingSelection, sendMessage])

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

    // Auto-scroll to bottom when new messages arrive or content changes
    useEffect(() => {
        scrollToBottom()
    }, [taskMessages, currentResponse, localResponse, gptResponse])

    // Auto-scroll to bottom when streaming starts or finishes
    useEffect(() => {
        if (isStreaming || (!isStreaming && (currentResponse || localResponse || gptResponse))) {
            // Small delay to ensure content is rendered before scrolling
            setTimeout(scrollToBottom, 100)
        }
    }, [isStreaming, currentResponse, localResponse, gptResponse])

    // Auto-scroll to bottom on initial load
    useEffect(() => {
        if (taskMessages.length > 0) {
            // Delay to ensure all content is rendered
            setTimeout(scrollToBottom, 300)
        }
    }, [task])

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

    // State for correction modal
    const [correctionModalOpen, setCorrectionModalOpen] = useState<boolean>(false)
    const [correctionMessageIndex, setCorrectionMessageIndex] = useState<number>(-1)

    // Handle learning feedback for assistant messages (SIMPLIFIED WORKFLOW)
    const handleLearningFeedback = async (type: string, message: TaskMessage, index: number) => {
        console.log('🔴 handleLearningFeedback called with type:', type, 'index:', index)

        if (!id) {
            console.log('❌ No task ID, returning')
            return
        }

        try {
            const buttonId = getButtonId(index, type)

            // Mark button as clicked
            setClickedButtons(prev => new Set(prev).add(buttonId))

            if (type === 'correct') {
                // Open correction modal
                console.log('🔧 Opening correction modal for message index:', index)
                setCorrectionMessageIndex(index)
                setCorrectionModalOpen(true)
                console.log('📱 Modal state set - correctionModalOpen: true, correctionMessageIndex:', index)

                // Get MCP server actions for this task
                if (task?.mcp_servers && task.mcp_servers.length > 0) {
                    console.log('🛠️ Getting MCP server actions for servers:', task.mcp_servers)
                    getMcpServerActions(task.mcp_servers)
                } else {
                    console.log('⚠️ No MCP servers found for task')
                }
                return // Don't send feedback yet, wait for correction submission
            }

            if (type === 'approve') {
                // Send approval feedback
                await sendLearningFeedback(type, message, id, undefined, index)

                // Show user confirmation
                console.log(`Learning feedback sent: ${type} for message at index ${index}`)
            }

        } catch (error) {
            console.error('Failed to send learning feedback:', error)
        }
    }

    // Handle current response learning feedback (SIMPLIFIED WORKFLOW)
    const handleCurrentResponseLearningFeedback = async (type: string) => {
        console.log('🔴 handleCurrentResponseLearningFeedback called with type:', type)

        if (!id || !currentResponse) {
            console.log('❌ No task ID or current response, returning')
            return
        }

        try {
            const buttonId = getCurrentButtonId(type)

            // Mark button as clicked
            setClickedButtons(prev => new Set(prev).add(buttonId))

            if (type === 'correct') {
                // Open correction modal for current response
                console.log('🔧 Opening correction modal for CURRENT response')
                setCorrectionMessageIndex(-1) // -1 indicates current response
                setCorrectionModalOpen(true)
                console.log('📱 Modal state set - correctionModalOpen: true, correctionMessageIndex: -1 (current response)')

                // Get MCP server actions for this task
                if (task?.mcp_servers && task.mcp_servers.length > 0) {
                    console.log('🛠️ Getting MCP server actions for servers:', task.mcp_servers)
                    getMcpServerActions(task.mcp_servers)
                } else {
                    console.log('⚠️ No MCP servers found for task')
                }
                return // Don't send feedback yet, wait for correction submission
            }

            if (type === 'approve') {
                const message = { role: 'assistant', content: currentResponse, timestamp: new Date().toISOString() }
                await sendLearningFeedback(type, message, id, undefined, -1) // -1 indicates current response

                // Show user confirmation
                console.log(`Learning feedback sent: ${type} for current response`)
            }

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

            await sendLearningFeedback(feedbackType!, messageToSend, id!, commentText, messageIndex >= 0 ? messageIndex : undefined)

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

    // Handle model selection
    const handleModelSelection = async (selectedModel: 'local' | 'gpt') => {
        if (!id || !isDualResponse) return

        try {
            console.log(`🐛 [Frontend] User selected ${selectedModel} model`)

            // Set processing flag to prevent auto-triggers
            setIsProcessingSelection(true)

            // Get the selected response content
            const selectedContent = selectedModel === 'local' ? localResponse : gptResponse

            console.log(`🐛 [Frontend] Selected content length: ${selectedContent.length}`)

            // Immediately add the selected response to task messages for instant UI feedback
            const newMessage = {
                role: 'assistant',
                content: selectedContent,
                timestamp: new Date().toISOString()
            }
            setTaskMessages(prev => [...prev, newMessage])
            console.log(`🐛 [Frontend] Added selected message to task messages`)

            // Clear the dual response state immediately
            clearDualResponse()
            console.log(`🐛 [Frontend] Cleared dual response state`)

            // Send the selection to the backend (will handle tool execution if needed)
            console.log(`🐛 [Frontend] About to call selectModel...`)
            await selectModel(selectedModel, id)
            console.log(`🐛 [Frontend] selectModel completed`)

            // Reload task messages after a short delay to get any additional updates (like tool execution results)
            setTimeout(async () => {
                try {
                    console.log(`🐛 [Frontend] Reloading task messages after delay...`)
                    const taskData = await getTask(id)
                    if (taskData && !taskData.error) {
                        setTaskMessages(taskData.messages || [])
                        console.log(`🐛 [Frontend] Task messages reloaded, count: ${taskData.messages?.length || 0}`)
                    }
                } catch (error) {
                    console.error('🐛 [Frontend] Error reloading task messages after selection:', error)
                } finally {
                    // Clear processing flag after reload completes
                    setIsProcessingSelection(false)
                    console.log(`🐛 [Frontend] Processing selection flag cleared`)
                }
            }, 1000)

        } catch (error) {
            console.error('🐛 [Frontend] Error selecting model:', error)
            setIsProcessingSelection(false)
        }
    }

    // Convert MCP server actions to tool format for CorrectionModal
    const getAvailableTools = () => {
        const tools: Array<{
            name: string
            description: string
            parameters: {
                type: string
                properties: Record<string, any>
                required: string[]
            }
        }> = []

        console.log('🔨 getAvailableTools called')
        console.log('📋 Task MCP servers:', task?.mcp_servers)
        console.log('🛠️ Available mcpServerActions:', mcpServerActions)

        if (task?.mcp_servers) {
            task.mcp_servers.forEach(serverId => {
                const actions = mcpServerActions[serverId] || []
                console.log(`🔧 Processing server ${serverId} with ${actions.length} actions`)

                actions.forEach(action => {
                    console.log(`🔍 Processing action: ${action.name}`)
                    console.log(`📋 Raw action data:`, action)
                    console.log(`📋 Action parameters:`, action.parameters)

                    let parameters = {
                        type: 'object',
                        properties: {},
                        required: []
                    }

                    if (action.parameters && typeof action.parameters === 'object') {
                        console.log(`📋 Found parameters object:`, action.parameters)
                        parameters = {
                            type: action.parameters.type || 'object',
                            properties: action.parameters.properties || {},
                            required: action.parameters.required || []
                        }
                        console.log(`📋 Final parameters:`, parameters)
                    } else {
                        console.log(`⚠️ No parameters found for action ${action.name}`)
                    }

                    const toolName = `${serverId}.${action.name}`
                    console.log(`➕ Adding tool: ${toolName}`)
                    tools.push({
                        name: toolName,
                        description: action.description,
                        parameters
                    })
                })
            })
        }

        console.log(`✅ getAvailableTools returning ${tools.length} tools:`, tools)
        return tools
    }

    // Handle correction modal submission
    const handleCorrectionSubmit = async (
        taskId: string,
        messageIndex: number,
        correctedToolCall: { name: string; arguments: Record<string, any> },
        thought: string,
        shouldExecute: boolean
    ) => {
        try {
            const result = await sendCorrectionWithExecution(
                taskId,
                messageIndex,
                correctedToolCall,
                thought,
                shouldExecute
            )

            // Close modal
            setCorrectionModalOpen(false)
            setCorrectionMessageIndex(-1)

            // If successful, reload task to see the corrected response
            if (result && !result.error) {
                setTimeout(async () => {
                    try {
                        const taskData = await getTask(taskId)
                        if (taskData && !taskData.error) {
                            setTaskMessages(taskData.messages || [])
                        }
                    } catch (error) {
                        console.error('Error reloading task after correction:', error)
                    }
                }, 500) // Reduced timeout since we just need to reload the corrected message
            }

            return result
        } catch (error) {
            console.error('Failed to submit correction:', error)
            return { error: 'Failed to submit correction' }
        }
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

    // Debug modal state
    console.log('🔧 About to render CorrectionModal with:', { correctionModalOpen, correctionMessageIndex, taskId: id })

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
                <div className="p-8 pt-20">
                    <div className="max-w-4xl mx-auto w-full">

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
                                    {taskMessages
                                        .filter((message) => {
                                            // Hide simple "approved" user messages
                                            if (message.role === 'user' && message.content.trim() === 'approved') {
                                                return false;
                                            }
                                            return true;
                                        })
                                        .map((message, index) => (
                                            <div
                                                key={index}
                                                className="text-base font-mono glow whitespace-pre-wrap leading-relaxed"
                                                style={{ marginBottom: '40px' }}
                                            >
                                                {message.role === 'user' ? (
                                                    <div className="text-cyan-300 mb-2">User</div>
                                                ) : message.role === 'tool' ? (
                                                    <div className="text-orange-300 mb-2">Tool Response</div>
                                                ) : (
                                                    <div className="text-green-300 mb-2">Assistant</div>
                                                )}
                                                <div className="border-l ml-2" style={{ paddingLeft: '20px', fontWeight: '300', color: '#E5E7EB', fontSize: '0.9rem', borderLeftColor: '#61FDFC', borderLeftWidth: '2px' }}>
                                                    {message.role === 'tool' ? (
                                                        <ToolResponseDisplay content={message.content} />
                                                    ) : (
                                                        parseContent(message.content)
                                                    )}
                                                </div>

                                                {/* Learning buttons for assistant messages - SIMPLIFIED */}
                                                {message.role === 'assistant' && (
                                                    <>
                                                        <div className="flex gap-4 mb-2" style={{ marginTop: '16px', marginLeft: '24px' }}>
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
                                                        </div>

                                                        {/* Inline comment box for historical messages */}
                                                        {activeCommentBox === getButtonId(index, 'comment') && (
                                                            <div style={{ marginTop: '16px', marginLeft: '24px' }}>
                                                                <textarea
                                                                    value={commentText}
                                                                    onChange={(e) => setCommentText(e.target.value)}
                                                                    placeholder="Add your comment..."
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
                                                        {submittedComments[getButtonId(index, 'comment')] && (
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
                                                                        COMMENT
                                                                    </div>
                                                                    <div style={{ color: '#E5E7EB' }}>
                                                                        {submittedComments[getButtonId(index, 'comment')]}
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

                        {/* Dual Response Display */}
                        {isDualResponse && (
                            <div className="text-base font-mono glow" style={{ marginBottom: '40px' }}>
                                <div className="text-green-300 mb-4">Choose the better action:</div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {/* Local Model Response */}
                                    <div
                                        className={`border rounded-lg p-4 cursor-pointer transition-all duration-300 ${localFinished
                                            ? 'border-glow border-gray-600'
                                            : 'border-gray-600'
                                            }`}
                                        onClick={() => localFinished && handleModelSelection('local')}
                                    >
                                        <div className="flex items-center gap-2 mb-3">
                                            <div className="w-3 h-3 rounded-full bg-blue-400"></div>
                                            <div className="text-sm text-gray-300">Action A</div>
                                        </div>
                                        <div className="border-l ml-2" style={{ paddingLeft: '20px', fontWeight: '300', color: '#E5E7EB', fontSize: '0.9rem', borderLeftColor: '#61FDFC', borderLeftWidth: '2px' }}>
                                            <span className="max-w-full break-words">
                                                {parseContentWithConsistentFormatting(localResponse)}
                                                {!localFinished && (
                                                    <motion.span
                                                        className="typing-cursor inline"
                                                        animate={{ opacity: [1, 0] }}
                                                        transition={{ duration: 1, repeat: Infinity }}
                                                    />
                                                )}
                                            </span>
                                        </div>
                                    </div>

                                    {/* GPT Model Response */}
                                    <div
                                        className={`border rounded-lg p-4 cursor-pointer transition-all duration-300 ${gptFinished
                                            ? 'border-glow border-gray-600'
                                            : 'border-gray-600'
                                            }`}
                                        onClick={() => gptFinished && handleModelSelection('gpt')}
                                    >
                                        <div className="flex items-center gap-2 mb-3">
                                            <div className="w-3 h-3 rounded-full bg-purple-400"></div>
                                            <div className="text-sm text-gray-300">Action B</div>
                                        </div>
                                        <div className="border-l ml-2" style={{ paddingLeft: '20px', fontWeight: '300', color: '#E5E7EB', fontSize: '0.9rem', borderLeftColor: '#61FDFC', borderLeftWidth: '2px' }}>
                                            <span className="max-w-full break-words">
                                                {parseContentWithConsistentFormatting(gptResponse)}
                                                {!gptFinished && (
                                                    <motion.span
                                                        className="typing-cursor inline"
                                                        animate={{ opacity: [1, 0] }}
                                                        transition={{ duration: 1, repeat: Infinity }}
                                                    />
                                                )}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Current Streaming Response (Single Model) */}
                        {!isDualResponse && (currentResponse || isStreaming) && (
                            <div
                                className="text-base font-mono glow whitespace-pre-wrap leading-relaxed word-break-normal max-w-full overflow-wrap-anywhere"
                                style={{ marginBottom: '40px' }}
                            >
                                <div className="relative">
                                    <div className="text-green-300 mb-2">Assistant:</div>
                                    <div className="border-l ml-2" style={{ paddingLeft: '20px', fontWeight: '300', color: '#E5E7EB', fontSize: '0.9rem', borderLeftColor: '#61FDFC', borderLeftWidth: '2px' }}>
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

                                    {/* Learning buttons for current response (only when not streaming) - SIMPLIFIED */}
                                    {!isStreaming && currentResponse && (
                                        <div className="flex gap-4 mb-2" style={{ marginTop: '16px', marginLeft: '24px' }}>
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
                                        </div>
                                    )}

                                    {/* Inline comment box for current response */}
                                    {activeCommentBox === getCurrentButtonId('comment') && (
                                        <div style={{ marginTop: '16px', marginLeft: '24px' }}>
                                            <textarea
                                                value={commentText}
                                                onChange={(e) => setCommentText(e.target.value)}
                                                placeholder="Add your comment..."
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

                                    {/* MCP Server Actions selection box for current response */}
                                    {showMcpActionsBox === getCurrentButtonId('correct') && (
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
                                                    onClick={() => handleMcpActionSelect(showMcpActionsBox!, -1)}
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

                        {/* Bottom reference for auto-scroll */}
                        <div ref={bottomRef} />

                        {/* Input Box - Always present with consistent spacing */}
                        <motion.div
                            className="max-w-2xl mx-auto"
                            style={{ marginTop: '60px' }}
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
                                        placeholder="Comment on the agents progress (type @ to mention MCP servers)"
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
                                        onApiKeyRequired={() => { }}
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

            {/* Correction Modal */}
            <CorrectionModal
                isOpen={correctionModalOpen}
                onClose={() => setCorrectionModalOpen(false)}
                availableTools={getAvailableTools()}
                currentTaskId={id || ''}
                messageIndex={correctionMessageIndex}
                onSubmitCorrection={handleCorrectionSubmit}
            />
        </div>
    )
}

export default TaskPage 