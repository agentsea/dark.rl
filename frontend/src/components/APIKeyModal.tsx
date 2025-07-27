import React, { useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import type { MCPServer } from '../hooks/useWebSocket'
import { setAPIKey } from '../lib/apiKeys'

interface APIKeyModalProps {
    isOpen: boolean;
    onClose: () => void;
    server: MCPServer | null;
    onSave: () => void;
}

export const APIKeyModal: React.FC<APIKeyModalProps> = ({ isOpen, onClose, server, onSave }) => {
    const [apiKeys, setApiKeys] = useState<Record<string, string>>({});
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [isBrowser, setIsBrowser] = useState(false)

    useEffect(() => {
        setIsBrowser(true)
    }, [])

    useEffect(() => {
        if (server) {
            const initialEnvVars: Record<string, string> = {}
            const allVars = [...(server.required_env || []), ...(server.optional_env || [])];
            allVars.forEach(key => {
                initialEnvVars[key] = ''
            })
            setApiKeys(initialEnvVars)
        }
    }, [server])

    const handleSave = async () => {
        if (!server) return

        const missingRequired = (server.required_env || []).some(key => !apiKeys[key]?.trim());

        if (missingRequired) {
            setError('Please fill in all required fields.')
            return
        }

        setLoading(true)
        setError('')

        try {
            for (const key in apiKeys) {
                if (apiKeys[key]?.trim()) {
                    await setAPIKey(key, apiKeys[key].trim())
                }
            }

            onSave()

            handleClose()
        } catch (err) {
            setError('Failed to save environment variables. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    const handleClose = () => {
        setApiKeys({})
        setError('')
        onClose()
    }

    const handleInputChange = (key: string, value: string) => {
        setApiKeys(prev => ({ ...prev, [key]: value }))
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !loading) {
            handleSave()
        }
        if (e.key === 'Escape') {
            handleClose()
        }
    }

    const modalContent = (
        <>
            {isOpen && server && (
                <div
                    className="fixed z-50 flex items-center justify-center"
                    style={{
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%'
                    }}
                >
                    <div
                        className="bg-black rounded-lg shadow-2xl backdrop-blur-sm p-6 w-full max-w-md mx-4 flex flex-col"
                        style={{
                            background: 'rgba(0, 0, 0, 0.95)',
                            border: '1px solid rgba(97, 253, 252, 0.3)',
                            boxShadow: '0 0 30px rgba(97, 253, 252, 0.3)',
                            maxHeight: '90vh',
                            width: '100%',
                            maxWidth: '500px',
                        }}
                    >
                        <div className="text-center mb-6 flex-shrink-0">
                            <h2 className="text-xl font-mono glow mb-2" style={{ color: '#61FDFC' }}>
                                Environment Variables Required
                            </h2>
                            <p className="text-sm font-mono" style={{ color: '#61FDFC', opacity: 0.8 }}>
                                @{server.id} requires configuration to function
                            </p>
                            <p className="text-xs font-mono mt-2" style={{ color: '#61FDFC', opacity: 0.6 }}>
                                {server.description}
                            </p>
                        </div>

                        <div
                            className="space-y-4 pr-2 flex-grow"
                            style={{ overflowY: 'auto', minHeight: 0 }}
                        >
                            {(server.required_env || []).map(key => (
                                <div key={key}>
                                    <label className="block text-sm font-mono mb-2" style={{ color: '#61FDFC' }}>
                                        {key} <span className="text-red-400">*</span>
                                    </label>
                                    <input
                                        type="password"
                                        value={apiKeys[key] || ''}
                                        onChange={(e) => handleInputChange(key, e.target.value)}
                                        onKeyPress={handleKeyPress}
                                        placeholder={`Enter ${key}...`}
                                        className="form-input"
                                        autoFocus={key === server.required_env[0]}
                                    />
                                </div>
                            ))}

                            {(server.optional_env || []).length > 0 && <hr style={{ borderColor: 'rgba(97, 253, 252, 0.3)' }} />}

                            {(server.optional_env || []).map(key => (
                                <div key={key}>
                                    <label className="block text-sm font-mono mb-2" style={{ color: '#61FDFC' }}>
                                        {key} <span className="opacity-60">(optional)</span>
                                    </label>
                                    <input
                                        type="password"
                                        value={apiKeys[key] || ''}
                                        onChange={(e) => handleInputChange(key, e.target.value)}
                                        onKeyPress={handleKeyPress}
                                        placeholder={`Enter ${key}...`}
                                        className="form-input"
                                    />
                                </div>
                            ))}

                            {error && (
                                <div className="text-red-400 text-xs font-mono mt-4">
                                    {error}
                                </div>
                            )}

                            <div className="text-xs font-mono mt-4" style={{ color: '#61FDFC', opacity: 0.6 }}>
                                ⚠️ Environment variables are stored locally in ~/.agentsea/api_keys.json
                            </div>
                        </div>

                        <div className="flex justify-end mt-6 flex-shrink-0" style={{ gap: '0.75rem' }}>
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
                                disabled={loading || (server.required_env || []).some(key => !apiKeys[key]?.trim())}
                                className="px-4 py-2 text-sm font-mono border rounded transition-colors disabled:opacity-50"
                                style={{
                                    borderColor: 'rgba(97, 253, 252, 0.3)',
                                    color: (loading || (server.required_env || []).some(key => !apiKeys[key]?.trim())) ? '#61FDFC' : '#000',
                                    backgroundColor: (loading || (server.required_env || []).some(key => !apiKeys[key]?.trim())) ? 'transparent' : '#61FDFC'
                                }}
                            >
                                {loading ? 'Saving...' : 'Save Configuration'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    )

    if (isBrowser) {
        return createPortal(modalContent, document.body)
    } else {
        return null
    }
} 