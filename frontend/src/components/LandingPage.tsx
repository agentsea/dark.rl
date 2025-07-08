import { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import TypingAnimation from './TypingAnimation'
import MCPServerDropdown from './MCPServerDropdown'
import APIKeyModal from './APIKeyModal'
import useWebSocket, { ConnectionStatus } from '../hooks/useWebSocket'
import type { MCPServer } from '../hooks/useWebSocket'

function LandingPage() {
    const [prompt, setPrompt] = useState('')
    const [showMCPDropdown, setShowMCPDropdown] = useState(false)
    const [mcpQuery, setMcpQuery] = useState('')
    const [cursorPosition, setCursorPosition] = useState(0)
    const [visibleExamples, setVisibleExamples] = useState<(number | null)[]>(Array(4).fill(null))
    const [exampleTimestamps, setExampleTimestamps] = useState<(number | null)[]>(Array(4).fill(null))
    const [hasInitiallyLoaded, setHasInitiallyLoaded] = useState(false)
    const [pulseTriggers, setPulseTriggers] = useState<number[]>(Array(4).fill(0))
    const [showAPIKeyModal, setShowAPIKeyModal] = useState(false)
    const [apiKeyServer, setApiKeyServer] = useState<MCPServer | null>(null)

    const navigate = useNavigate()

    const examples = useMemo(() => [
        "Use @playwright to find the best restaurants in Boulder and put them in @GoogleDocs",
        "Help me @research quantum computing trends and @summarize them",
        "Use @calculator to compute my monthly budget and @email the results",
        "@translate this document to Spanish and @save it to my desktop",
        "Find @weather forecast for next week and @schedule my outdoor activities",
        "@scrape product prices from Amazon and @compare them in a spreadsheet",
        "Use @github to find trending AI repositories and @bookmark the top 10",
        "@analyze my stock portfolio performance and @generate a report"
    ], [])

    const {
        connectionStatus,
        mcpServers,
        loadingMcpServers,
        connect,
        getMcpServers,
        createTask
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

    // Initial loading sequence - wait 1 second then fade in all examples together
    useEffect(() => {
        if (connectionStatus !== ConnectionStatus.CONNECTED || hasInitiallyLoaded) return

        const initialLoadTimeout = setTimeout(() => {
            // Add all 4 examples at once - pick 4 random unique indices
            const currentTime = Date.now()
            const availableIndices = examples.map((_: string, index: number) => index) // [0, 1, 2, 3, 4, 5, 6, 7]
            const selectedIndices = []

            // Pick 4 unique random indices
            for (let i = 0; i < 4; i++) {
                const randomIndex = Math.floor(Math.random() * availableIndices.length)
                selectedIndices.push(availableIndices[randomIndex])
                availableIndices.splice(randomIndex, 1) // Remove selected index to avoid duplicates
            }

            setVisibleExamples(selectedIndices)
            setExampleTimestamps([currentTime, currentTime, currentTime, currentTime])

            // Mark initial loading as complete after examples fade in
            // All examples start at the same time, take 2.5s to fade in, plus 1s buffer = 4.5s total
            setTimeout(() => setHasInitiallyLoaded(true), 4500) // Start pulsing after all are visible
        }, 1000) // Wait 1 second before starting

        return () => clearTimeout(initialLoadTimeout)
    }, [connectionStatus, hasInitiallyLoaded, examples])

    // Individual pulse animations for examples
    useEffect(() => {
        if (connectionStatus !== ConnectionStatus.CONNECTED || !hasInitiallyLoaded) return

        const scheduleRandomPulse = () => {
            // Random delay between 2-8 seconds
            const delay = Math.random() * 6000 + 2000

            setTimeout(() => {
                // Pick a random example that exists
                const availableIndexes = visibleExamples
                    .map((example: number | null, index: number) => example !== null ? index : -1)
                    .filter((index: number) => index !== -1)

                if (availableIndexes.length > 0) {
                    const randomIndex = availableIndexes[Math.floor(Math.random() * availableIndexes.length)]

                    // Trigger pulse for this example
                    setPulseTriggers(prev => {
                        const newTriggers = [...prev]
                        newTriggers[randomIndex] = Date.now() // Use timestamp as trigger key
                        return newTriggers
                    })
                }

                // Schedule the next pulse
                scheduleRandomPulse()
            }, delay)
        }

        // Start the pulse scheduling
        scheduleRandomPulse()

        // Cleanup function isn't needed since we want this to run continuously
    }, [connectionStatus, hasInitiallyLoaded, visibleExamples])

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
        }

        setShowMCPDropdown(false)
    }

    // Handle API key required
    const handleApiKeyRequired = (server: MCPServer) => {
        setApiKeyServer(server)
        setShowAPIKeyModal(true)
        setShowMCPDropdown(false)
    }

    // Handle API key modal save
    const handleApiKeyModalSave = async (server: MCPServer, apiKey: string) => {
        // API key is already saved by the modal
        // Now we can select the server
        handleMCPServerSelect(server)
        // Refresh the MCP servers to get updated API key status
        getMcpServers(mcpQuery)
    }

    // Handle API key modal close
    const handleApiKeyModalClose = () => {
        setShowAPIKeyModal(false)
        setApiKeyServer(null)
    }

    // Close MCP dropdown
    const closeMCPDropdown = () => {
        setShowMCPDropdown(false)
    }

    // Handle example click
    const handleExampleClick = (exampleText: string) => {
        setPrompt(exampleText)
    }

    // Update MCP search when servers list changes
    useEffect(() => {
        if (showMCPDropdown && connectionStatus === ConnectionStatus.CONNECTED) {
            getMcpServers(mcpQuery)
        }
    }, [mcpQuery, showMCPDropdown, connectionStatus, getMcpServers])

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault()

        if (!prompt.trim() || connectionStatus !== ConnectionStatus.CONNECTED) {
            return
        }

        // Create task and navigate to task page
        const taskId = await createTask(prompt.trim(), 'qwen2.5-vl')

        if (taskId) {
            navigate(`/tasks/${taskId}`)
        }
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            const form = e.currentTarget.closest('form') as HTMLFormElement
            if (form) {
                // Use requestSubmit() instead of dispatchEvent to properly trigger form validation and onSubmit
                form.requestSubmit()
            }
        }
    }

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

            {/* LANDING PAGE CONTENT */}
            {connectionStatus === ConnectionStatus.CONNECTED ? (
                <div className="min-h-screen flex flex-col p-8">
                    {/* Logo positioned toward the top */}
                    <motion.div
                        className="flex justify-center pt-16 pb-8"
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <img
                            src="https://storage.googleapis.com/guisurfer-assets/dark_rl_extreme_glow.png"
                            alt="DARK.RL"
                            className="h-[300px] w-[300px] object-contain"
                        />
                    </motion.div>

                    {/* Text and input positioned in remaining space */}
                    <div className="flex-1 flex justify-center" style={{ alignItems: 'flex-start', paddingTop: '80px' }}>
                        <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>

                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.8, delay: 0.5 }}
                            >
                                <TypingAnimation
                                    text="What would you like me to learn today?"
                                    speed={40}
                                    className="text-2xl glow font-mono"
                                    onComplete={() => { }}
                                />
                            </motion.div>

                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.8, delay: 1.2 }}
                                className="mx-auto"
                                style={{ marginTop: '40px', width: '800px', maxWidth: '90vw', marginLeft: 'auto', marginRight: 'auto' }}
                            >
                                <form onSubmit={handleSubmit} className="space-y-4" style={{ width: '100%' }}>
                                    <div className="relative">
                                        <textarea
                                            value={prompt}
                                            onChange={handleTextareaChange}
                                            onKeyPress={handleKeyPress}
                                            className="form-textarea min-h-[120px] w-full glow resize-none font-mono text-lg"
                                            placeholder="Enter your learning objective... (type @ to mention MCP servers)"
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
                                            onApiKeyRequired={handleApiKeyRequired}
                                        />

                                        {/* API Key Modal */}
                                        <APIKeyModal
                                            isVisible={showAPIKeyModal}
                                            server={apiKeyServer}
                                            onClose={handleApiKeyModalClose}
                                            onSave={handleApiKeyModalSave}
                                        />
                                    </div>
                                </form>

                                {/* Floating Examples - Table Layout */}
                                <div style={{ marginTop: '100px' }}>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gridTemplateRows: '80px', gap: '20px' }}>
                                        {Array.from({ length: 4 }, (_, cellIndex) => {
                                            // Assign examples to static positions (no more random changes)
                                            const assignedExampleIndex = visibleExamples[cellIndex]
                                            const hasExample = assignedExampleIndex !== null && assignedExampleIndex !== undefined

                                            return (
                                                <div key={cellIndex} className="relative">
                                                    {hasExample && assignedExampleIndex !== null && (
                                                        <motion.div
                                                            key={`example-${cellIndex}`}
                                                            className="text-xs font-mono cursor-pointer w-full h-full"
                                                            initial={{ opacity: 0, scale: 0.8 }}
                                                            animate={{ opacity: 1, scale: 1 }}
                                                            transition={{ duration: 2.5, ease: "easeInOut" }}
                                                            onClick={() => handleExampleClick(examples[assignedExampleIndex])}
                                                            whileHover={{ scale: 1.05 }}
                                                            whileTap={{ scale: 0.95 }}
                                                        >
                                                            <motion.div
                                                                className="rounded-full border backdrop-blur-sm text-center w-full h-full"
                                                                animate={{
                                                                    borderColor: pulseTriggers[cellIndex] > 0 ? [
                                                                        'rgba(97, 253, 252, 0.3)',
                                                                        'rgba(97, 253, 252, 0.6)',
                                                                        'rgba(97, 253, 252, 0.3)'
                                                                    ] : 'rgba(97, 253, 252, 0.3)',
                                                                    backgroundColor: pulseTriggers[cellIndex] > 0 ? [
                                                                        'rgba(97, 253, 252, 0.1)',
                                                                        'rgba(97, 253, 252, 0.2)',
                                                                        'rgba(97, 253, 252, 0.1)'
                                                                    ] : 'rgba(97, 253, 252, 0.1)',
                                                                    boxShadow: pulseTriggers[cellIndex] > 0 ? [
                                                                        '0 0 10px rgba(97, 253, 252, 0.2)',
                                                                        '0 0 20px rgba(97, 253, 252, 0.4)',
                                                                        '0 0 10px rgba(97, 253, 252, 0.2)'
                                                                    ] : '0 0 10px rgba(97, 253, 252, 0.2)'
                                                                }}
                                                                transition={{
                                                                    duration: 1.2,
                                                                    ease: "easeInOut"
                                                                }}
                                                                style={{
                                                                    color: '#61FDFC',
                                                                    display: 'flex',
                                                                    alignItems: 'center',
                                                                    justifyContent: 'center',
                                                                    lineHeight: '1.3',
                                                                    padding: '12px'
                                                                }}
                                                            >
                                                                "{examples[assignedExampleIndex]}"
                                                            </motion.div>
                                                        </motion.div>
                                                    )}
                                                </div>
                                            )
                                        })}
                                    </div>
                                </div>
                            </motion.div>

                        </div>
                    </div>
                </div>
            ) : (
                // CONNECTION REQUIRED - ALSO CENTERED
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

export default LandingPage 