import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { MCPServer } from '../hooks/useWebSocket'
import { setAPIKey } from '../lib/apiKeys'

interface APIKeyModalProps {
    isVisible: boolean
    server: MCPServer | null
    onClose: () => void
    onSave: (server: MCPServer, apiKey: string) => void
}

export default function APIKeyModal({
    isVisible,
    server,
    onClose,
    onSave
}: APIKeyModalProps) {
    const [apiKey, setApiKey] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const handleSave = async () => {
        if (!server || !apiKey.trim()) {
            setError('Please enter a valid API key')
            return
        }

        setLoading(true)
        setError('')

        try {
            // Save API key to local storage
            if (server.api_key_env) {
                await setAPIKey(server.api_key_env, apiKey.trim())
            }

            // Call the onSave callback
            onSave(server, apiKey.trim())

            // Reset form
            setApiKey('')
            onClose()
        } catch (err) {
            setError('Failed to save API key. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    const handleClose = () => {
        setApiKey('')
        setError('')
        onClose()
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !loading) {
            handleSave()
        }
        if (e.key === 'Escape') {
            handleClose()
        }
    }

    return (
        <AnimatePresence>
            {isVisible && server && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-50 flex items-center justify-center"
                    style={{ backgroundColor: 'rgba(0, 0, 0, 0.8)' }}
                >
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.9, opacity: 0 }}
                        className="bg-black rounded-lg shadow-2xl backdrop-blur-sm p-6 w-full max-w-md mx-4"
                        style={{
                            background: 'rgba(0, 0, 0, 0.95)',
                            border: '1px solid rgba(97, 253, 252, 0.3)',
                            boxShadow: '0 0 30px rgba(97, 253, 252, 0.3)'
                        }}
                    >
                        <div className="text-center mb-6">
                            <h2 className="text-xl font-mono glow mb-2" style={{ color: '#61FDFC' }}>
                                API Key Required
                            </h2>
                            <p className="text-sm font-mono" style={{ color: '#61FDFC', opacity: 0.8 }}>
                                @{server.id} requires an API key to function
                            </p>
                            <p className="text-xs font-mono mt-2" style={{ color: '#61FDFC', opacity: 0.6 }}>
                                {server.description}
                            </p>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-mono mb-2" style={{ color: '#61FDFC' }}>
                                    {server.api_key_env} API Key:
                                </label>
                                <input
                                    type="password"
                                    value={apiKey}
                                    onChange={(e) => setApiKey(e.target.value)}
                                    onKeyPress={handleKeyPress}
                                    placeholder="Enter your API key..."
                                    className="w-full px-3 py-2 bg-black border rounded font-mono text-sm"
                                    style={{
                                        borderColor: 'rgba(97, 253, 252, 0.3)',
                                        color: '#61FDFC',
                                        backgroundColor: 'rgba(0, 0, 0, 0.8)'
                                    }}
                                    autoFocus
                                />
                            </div>

                            {error && (
                                <div className="text-red-400 text-xs font-mono">
                                    {error}
                                </div>
                            )}

                            <div className="text-xs font-mono" style={{ color: '#61FDFC', opacity: 0.6 }}>
                                ⚠️ API keys are stored locally in ~/.agentsea/api_keys.json
                            </div>
                        </div>

                        <div className="flex justify-end space-x-3 mt-6">
                            <button
                                onClick={handleClose}
                                className="px-4 py-2 text-sm font-mono border rounded transition-colors"
                                style={{
                                    borderColor: 'rgba(97, 253, 252, 0.3)',
                                    color: '#61FDFC',
                                    backgroundColor: 'transparent'
                                }}
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleSave}
                                disabled={loading || !apiKey.trim()}
                                className="px-4 py-2 text-sm font-mono border rounded transition-colors disabled:opacity-50"
                                style={{
                                    borderColor: 'rgba(97, 253, 252, 0.3)',
                                    color: loading || !apiKey.trim() ? '#61FDFC' : '#000',
                                    backgroundColor: loading || !apiKey.trim() ? 'transparent' : '#61FDFC'
                                }}
                            >
                                {loading ? 'Saving...' : 'Save API Key'}
                            </button>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    )
} 