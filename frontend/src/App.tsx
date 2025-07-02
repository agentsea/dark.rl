import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import TypingAnimation from './components/TypingAnimation'
import MCPServerDropdown from './components/MCPServerDropdown'
import useWebSocket, { ConnectionStatus } from './hooks/useWebSocket'
import type { MCPServer } from './hooks/useWebSocket'

function App() {
  const [prompt, setPrompt] = useState('')
  const [hasStarted, setHasStarted] = useState(false)
  const [showMCPDropdown, setShowMCPDropdown] = useState(false)
  const [mcpQuery, setMcpQuery] = useState('')
  const [cursorPosition, setCursorPosition] = useState(0)
  const [visibleExamples, setVisibleExamples] = useState<(number | null)[]>(Array(4).fill(null))
  const [exampleTimestamps, setExampleTimestamps] = useState<(number | null)[]>(Array(4).fill(null))
  const [lastUsedColumn, setLastUsedColumn] = useState(-1)
  const [hasInitiallyLoaded, setHasInitiallyLoaded] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const examples = [
    "Use @playwright to find the best restaurants in Boulder and put them in @GoogleDocs",
    "Help me @research quantum computing trends and @summarize them",
    "Use @calculator to compute my monthly budget and @email the results",
    "@translate this document to Spanish and @save it to my desktop",
    "Find @weather forecast for next week and @schedule my outdoor activities",
    "@scrape product prices from Amazon and @compare them in a spreadsheet",
    "Use @github to find trending AI repositories and @bookmark the top 10",
    "@analyze my stock portfolio performance and @generate a report"
  ]

  const {
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
  } = useWebSocket({
    url: 'ws://localhost:8000',
    autoConnect: true,  // Auto-connect on startup
    reconnectAttempts: 3,
    reconnectInterval: 3000
  })

  // Auto-connect to qwen2.5-vl when component mounts
  useEffect(() => {
    connect()
  }, [connect])

  // Initial loading sequence - wait 1 second then slowly fade in examples
  useEffect(() => {
    if (connectionStatus !== ConnectionStatus.CONNECTED || hasStarted || hasInitiallyLoaded) return

    const initialLoadTimeout = setTimeout(() => {
      let columnIndex = 0

      // Add examples one by one with delays
      const addInitialExample = () => {
        if (columnIndex < 3) { // Add up to 3 initial examples
          const currentTime = Date.now() // Get fresh timestamp for each example

          setVisibleExamples(prev => {
            const newGrid = [...prev]
            const availableExamples = examples
              .map((_, index) => index)
              .filter(index => !newGrid.includes(index))

            if (availableExamples.length > 0) {
              const randomExample = availableExamples[Math.floor(Math.random() * availableExamples.length)]
              newGrid[columnIndex] = randomExample
              return newGrid
            }
            return prev
          })

          setExampleTimestamps(prev => {
            const newTimestamps = [...prev]
            newTimestamps[columnIndex] = currentTime // Each example gets its own timestamp
            return newTimestamps
          })

          setLastUsedColumn(columnIndex)
          columnIndex++

          // Schedule next example after 1.5 seconds
          if (columnIndex < 3) {
            setTimeout(addInitialExample, 1500)
          } else {
            // Mark initial loading as complete after enough time for all to fade in
            setTimeout(() => setHasInitiallyLoaded(true), 4000) // 4 seconds to ensure all are fully visible
          }
        }
      }

      addInitialExample()
    }, 1000) // Wait 1 second before starting

    return () => clearTimeout(initialLoadTimeout)
  }, [connectionStatus, hasStarted, hasInitiallyLoaded, examples])

  // Manage example animations
  useEffect(() => {
    if (connectionStatus !== ConnectionStatus.CONNECTED || hasStarted || !hasInitiallyLoaded) return

    const interval = setInterval(() => {
      const currentTime = Date.now()

      setVisibleExamples(prevGrid => {
        const newGrid = [...prevGrid]

        // Remove old examples with timestamps check
        setExampleTimestamps(prevTimestamps => {
          const newTimestamps = [...prevTimestamps]

          // Remove old examples (only after they've been FULLY visible for a while)
          for (let i = 0; i < newGrid.length; i++) {
            if (newGrid[i] !== null && newTimestamps[i] !== null) {
              const timeVisible = currentTime - newTimestamps[i]!
              // Minimum time = fade-in duration (2.5s) + minimum display time (6s) = 8.5s minimum
              // During initial load, even longer protection
              const minVisibleTime = hasInitiallyLoaded ? 8500 : 12000
              if (timeVisible >= minVisibleTime && Math.random() < 0.25) { // Reduced chance from 30% to 25%
                newGrid[i] = null
                newTimestamps[i] = null
              }
            }
          }

          // Add new examples to columns, cycling through them
          const numToAdd = Math.floor(Math.random() * 2) + 1 // 1-2 examples

          for (let i = 0; i < numToAdd; i++) {
            const usedExamples = newGrid.filter(x => x !== null) as number[]
            const availableExamples = examples
              .map((_, index) => index)
              .filter(index => !usedExamples.includes(index))

            if (availableExamples.length > 0) {
              // Try to find next empty column in sequence
              let nextColumn = -1
              for (let j = 0; j < 4; j++) {
                const candidateColumn = (lastUsedColumn + 1 + j) % 4
                if (newGrid[candidateColumn] === null) {
                  nextColumn = candidateColumn
                  break
                }
              }

              // If no empty columns in sequence, pick any empty one
              if (nextColumn === -1) {
                const emptyCells = []
                for (let j = 0; j < newGrid.length; j++) {
                  if (newGrid[j] === null) {
                    emptyCells.push(j)
                  }
                }
                if (emptyCells.length > 0) {
                  nextColumn = emptyCells[Math.floor(Math.random() * emptyCells.length)]
                }
              }

              if (nextColumn !== -1) {
                const randomExample = availableExamples[Math.floor(Math.random() * availableExamples.length)]
                newGrid[nextColumn] = randomExample
                newTimestamps[nextColumn] = currentTime
                setLastUsedColumn(nextColumn)
              }
            }
          }

          return newTimestamps
        })

        return newGrid
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [connectionStatus, hasStarted, hasInitiallyLoaded, examples])

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
  const handleMCPServerSelect = (server: MCPServer) => {
    const atIndex = prompt.lastIndexOf('@', cursorPosition - 1)
    if (atIndex !== -1) {
      const beforeAt = prompt.substring(0, atIndex)
      const afterCursor = prompt.substring(cursorPosition)
      const newPrompt = `${beforeAt}@${server.id} ${afterCursor}`
      setPrompt(newPrompt)

      // Focus textarea and position cursor after the inserted server
      setTimeout(() => {
        if (textareaRef.current) {
          const newCursorPos = atIndex + server.id.length + 2
          textareaRef.current.focus()
          textareaRef.current.setSelectionRange(newCursorPos, newCursorPos)
        }
      }, 0)
    }

    setShowMCPDropdown(false)
  }

  // Close MCP dropdown
  const closeMCPDropdown = () => {
    setShowMCPDropdown(false)
  }

  // Handle example click
  const handleExampleClick = (exampleText: string) => {
    setPrompt(exampleText)
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  // Update MCP search when servers list changes
  useEffect(() => {
    if (showMCPDropdown && connectionStatus === ConnectionStatus.CONNECTED) {
      getMcpServers(mcpQuery)
    }
  }, [mcpQuery, showMCPDropdown, connectionStatus, getMcpServers])

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!prompt.trim() || connectionStatus !== ConnectionStatus.CONNECTED) {
      return
    }

    sendMessage(prompt.trim(), 'qwen2.5-vl')  // Always use qwen2.5-vl
    setPrompt('')
    setHasStarted(true)
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

      {/* SIMPLE CONDITIONAL LAYOUT */}
      {connectionStatus === ConnectionStatus.CONNECTED && !hasStarted ? (
        // INITIAL CENTERED STATE
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
                      ref={textareaRef}
                      value={prompt}
                      onChange={handleTextareaChange}
                      onKeyPress={handleKeyPress}
                      className="form-textarea min-h-[120px] w-full glow resize-none font-mono text-lg"
                      placeholder="Enter your learning objective... (type @ to mention MCP servers)"
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

                  {isStreaming && (
                    <div className="flex items-center justify-center gap-2 glow" style={{ color: '#61FDFC' }}>
                      <motion.div
                        className="w-2 h-2 bg-current rounded-full"
                        animate={{ opacity: [1, 0.3, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      />
                      <span className="text-sm font-mono">Processing neural input...</span>
                    </div>
                  )}
                </form>

                {/* Floating Examples - Table Layout */}
                <div style={{ marginTop: '100px' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gridTemplateRows: '80px', gap: '20px' }}>
                    {Array.from({ length: 4 }, (_, cellIndex) => {
                      // Assign examples to random cells
                      const assignedExampleIndex = visibleExamples[cellIndex]
                      const hasExample = assignedExampleIndex !== null && assignedExampleIndex !== undefined

                      return (
                        <div key={cellIndex} className="relative">
                          <AnimatePresence mode="wait">
                            {hasExample && assignedExampleIndex !== null && (
                              <motion.div
                                key={`${assignedExampleIndex}-${cellIndex}`}
                                className="text-xs font-mono cursor-pointer w-full h-full"
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.8 }}
                                transition={{ duration: 2.5, ease: "easeInOut" }}
                                onClick={() => handleExampleClick(examples[assignedExampleIndex])}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                              >
                                <div
                                  className="rounded-full border backdrop-blur-sm text-center w-full h-full"
                                  style={{
                                    backgroundColor: 'rgba(97, 253, 252, 0.1)',
                                    borderColor: 'rgba(97, 253, 252, 0.3)',
                                    color: '#61FDFC',
                                    boxShadow: '0 0 10px rgba(97, 253, 252, 0.2)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    lineHeight: '1.3',
                                    padding: '12px',
                                    animation: 'subtle-pulse 3s ease-in-out infinite'
                                  }}
                                >
                                  "{examples[assignedExampleIndex]}"
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </motion.div>

            </div>
          </div>
        </div>
      ) : connectionStatus === ConnectionStatus.CONNECTED ? (
        // AFTER INTERACTION STARTED
        <div className="min-h-screen p-8 pt-16">
          <div className="max-w-4xl mx-auto space-y-6">
            <motion.div
              className="flex justify-center"
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

            <motion.div
              className="max-w-2xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="relative">
                  <textarea
                    ref={textareaRef}
                    value={prompt}
                    onChange={handleTextareaChange}
                    onKeyPress={handleKeyPress}
                    className="form-textarea min-h-[120px] w-full glow resize-none font-mono text-lg"
                    placeholder="Enter your learning objective... (type @ to mention MCP servers)"
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

                {isStreaming && (
                  <div className="flex items-center justify-center gap-2 glow" style={{ color: '#61FDFC' }}>
                    <motion.div
                      className="w-2 h-2 bg-current rounded-full"
                      animate={{ opacity: [1, 0.3, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                    />
                    <span className="text-sm font-mono">Processing neural input...</span>
                  </div>
                )}
              </form>
            </motion.div>

            {/* Current Response Display */}
            {(currentResponse || isStreaming) && (
              <motion.div
                className="mt-12"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title glow">Neural Response</h3>
                  </div>
                  <div className="card-content">
                    <div className="min-h-[100px] p-4 bg-gray-50 rounded-md overflow-y-auto max-h-[400px]">
                      {currentResponse ? (
                        <TypingAnimation
                          text={currentResponse}
                          speed={20}
                          className="text-sm glow font-mono whitespace-pre-wrap"
                        />
                      ) : (
                        <div className="text-sm text-gray-600 glow font-mono">
                          Generating response...
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Conversation History */}
            {messages.length > 0 && (
              <motion.div
                className="mt-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title glow">Learning History</h3>
                    <button
                      onClick={clearMessages}
                      className="text-xs px-2 py-1 border rounded font-mono"
                      style={{
                        borderColor: 'rgba(97, 253, 252, 0.3)',
                        color: '#61FDFC'
                      }}
                    >
                      Clear
                    </button>
                  </div>
                  <div className="card-content">
                    <div className="space-y-4 max-h-[300px] overflow-y-auto">
                      {messages.map((message, index) => (
                        <motion.div
                          key={index}
                          className={`p-3 rounded-md border`}
                          style={{
                            borderColor: 'rgba(97, 253, 252, 0.3)',
                            backgroundColor: message.role === 'user' ? 'rgba(97, 253, 252, 0.1)' : 'rgba(97, 253, 252, 0.05)'
                          }}
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.3, delay: index * 0.1 }}
                        >
                          <div className={`text-xs mb-2 font-mono glow`} style={{ color: '#61FDFC' }}>
                            {message.role === 'user' ? '[USER]' : '[AI]'}
                            {message.timestamp && (
                              <span className="ml-2 opacity-60">
                                {new Date(message.timestamp).toLocaleTimeString()}
                              </span>
                            )}
                          </div>
                          <div className={`text-sm font-mono whitespace-pre-wrap glow`} style={{ color: '#61FDFC' }}>
                            {message.content}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
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

export default App
