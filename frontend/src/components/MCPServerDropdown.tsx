import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { MCPServer } from '../hooks/useWebSocket'

interface MCPServerDropdownProps {
    isVisible: boolean
    servers: MCPServer[]
    loading: boolean
    onSelect: (server: MCPServer) => void
    onClose: () => void
    query: string
    onApiKeyRequired: (server: MCPServer) => void
}

export default function MCPServerDropdown({
    isVisible,
    servers,
    loading,
    onSelect,
    onClose,
    query,
    onApiKeyRequired
}: MCPServerDropdownProps) {
    const [selectedIndex, setSelectedIndex] = useState(0)

    // Reset selected index when servers change
    useEffect(() => {
        setSelectedIndex(0)
    }, [servers])

    // Handle server selection with API key check
    const handleServerSelect = (server: MCPServer) => {
        // Check if server requires an API key and doesn't have one
        if (server.requires_api_key && !server.api_key_available) {
            onApiKeyRequired(server)
        } else {
            onSelect(server)
        }
    }

    // Handle keyboard navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!isVisible) return

            switch (e.key) {
                case 'Escape':
                    onClose()
                    break
                case 'ArrowDown':
                    e.preventDefault()
                    setSelectedIndex(prev => (prev + 1) % servers.length)
                    break
                case 'ArrowUp':
                    e.preventDefault()
                    setSelectedIndex(prev => (prev - 1 + servers.length) % servers.length)
                    break
                case 'Enter':
                    e.preventDefault()
                    if (servers[selectedIndex]) {
                        handleServerSelect(servers[selectedIndex])
                    }
                    break
            }
        }

        if (isVisible) {
            document.addEventListener('keydown', handleKeyDown)
            return () => document.removeEventListener('keydown', handleKeyDown)
        }
    }, [isVisible, onClose, servers, selectedIndex, onSelect])

    return (
        <AnimatePresence>
            {isVisible && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                    className="absolute z-50 mt-2 bg-black rounded-lg shadow-2xl backdrop-blur-sm"
                    style={{
                        background: 'rgba(0, 0, 0, 0.95)',
                        border: '1px solid rgba(97, 253, 252, 0.3)',
                        boxShadow: '0 0 20px rgba(97, 253, 252, 0.2)',
                        width: '90%',
                        maxWidth: '600px'
                    }}
                >
                    {loading ? (
                        <div className="p-4 text-center font-mono" style={{ color: '#61FDFC' }}>
                            <div className="inline-block animate-pulse">Searching servers...</div>
                        </div>
                    ) : servers.length === 0 ? (
                        <div className="p-4 text-center font-mono" style={{ color: '#61FDFC', opacity: 0.7 }}>
                            {query ? `No servers found for "${query}"` : 'No MCP servers available'}
                        </div>
                    ) : (
                        <div className="max-h-64 overflow-y-auto">
                            {servers.map((server, index) => {
                                const isSelected = index === selectedIndex
                                return (
                                    <motion.div
                                        key={server.id}
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ duration: 0.2, delay: index * 0.05 }}
                                        onClick={() => handleServerSelect(server)}
                                        onMouseEnter={() => setSelectedIndex(index)}
                                        className={`p-3 cursor-pointer last:border-b-0 transition-all duration-200`}
                                        style={{
                                            borderBottom: '1px solid rgba(97, 253, 252, 0.2)',
                                            backgroundColor: isSelected ? 'rgba(97, 253, 252, 0.25)' : undefined,
                                            boxShadow: isSelected ? '0 0 15px rgba(97, 253, 252, 0.4)' : undefined
                                        }}
                                        whileHover={{
                                            scale: 1.02,
                                            boxShadow: '0 0 12px rgba(97, 253, 252, 0.3)'
                                        }}
                                        whileTap={{ scale: 0.98 }}
                                    >
                                        <div className="flex items-start justify-between">
                                            <div className="flex-1">
                                                <div className="flex items-center gap-2">
                                                    <div className="font-mono font-semibold transition-colors" style={{ color: '#61FDFC' }}>
                                                        @{server.id}
                                                    </div>
                                                    {server.requires_api_key && (
                                                        <span
                                                            className="text-xs font-mono px-1 py-0.5 rounded"
                                                            style={{
                                                                backgroundColor: server.api_key_available
                                                                    ? 'rgba(34, 197, 94, 0.2)'
                                                                    : 'rgba(239, 68, 68, 0.2)',
                                                                color: server.api_key_available
                                                                    ? '#22c55e'
                                                                    : '#ef4444',
                                                                border: `1px solid ${server.api_key_available
                                                                    ? 'rgba(34, 197, 94, 0.3)'
                                                                    : 'rgba(239, 68, 68, 0.3)'}`
                                                            }}
                                                        >
                                                            {server.api_key_available ? '🔓 KEY' : '🔐 NO KEY'}
                                                        </span>
                                                    )}
                                                </div>
                                                <div className="text-sm font-mono mt-1 transition-colors" style={{ color: '#61FDFC', opacity: 0.8 }}>
                                                    {server.name}
                                                </div>
                                                <div className="text-xs font-mono mt-1 transition-colors" style={{ color: '#61FDFC', opacity: 0.6 }}>
                                                    {server.description}
                                                </div>
                                                {server.requires_api_key && !server.api_key_available && (
                                                    <div className="text-xs font-mono mt-1" style={{ color: '#ef4444', opacity: 0.8 }}>
                                                        Click to configure {server.api_key_env} API key
                                                    </div>
                                                )}
                                            </div>
                                            <div className="text-xs font-mono ml-2 transition-colors" style={{ color: '#61FDFC' }}>
                                                ↵
                                            </div>
                                        </div>
                                    </motion.div>
                                )
                            })}
                        </div>
                    )}

                    {/* Footer hint */}
                    <div className="p-2 text-xs font-mono text-center" style={{
                        borderTop: '1px solid rgba(97, 253, 252, 0.2)',
                        color: '#61FDFC',
                        opacity: 0.7
                    }}>
                        ↑↓ to navigate • ↵ to select • ESC to close
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    )
} 