import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface Tool {
    name: string
    description: string
    parameters: {
        [key: string]: {
            type: string
            description: string
            required: boolean
        }
    }
}

interface ToolCallModalProps {
    isVisible: boolean
    tools: Tool[]
    suggestedTool?: string
    thinking?: string
    onExecute: (toolName: string, parameters: Record<string, any>) => void
    onCancel: () => void
}

export default function ToolCallModal({
    isVisible,
    tools,
    suggestedTool,
    thinking,
    onExecute,
    onCancel
}: ToolCallModalProps) {
    const [selectedTool, setSelectedTool] = useState<string>('')
    const [parameters, setParameters] = useState<Record<string, any>>({})
    const [validationErrors, setValidationErrors] = useState<Record<string, string>>({})

    // Set default tool when modal opens
    useEffect(() => {
        if (isVisible && suggestedTool && tools.find(t => t.name === suggestedTool)) {
            setSelectedTool(suggestedTool)
        } else if (isVisible && tools.length > 0) {
            setSelectedTool(tools[0].name)
        }
    }, [isVisible, suggestedTool, tools])

    // Reset parameters when tool changes
    useEffect(() => {
        if (selectedTool) {
            const tool = tools.find(t => t.name === selectedTool)
            if (tool) {
                const newParams: Record<string, any> = {}
                Object.keys(tool.parameters).forEach(paramName => {
                    newParams[paramName] = ''
                })
                setParameters(newParams)
                setValidationErrors({})
            }
        }
    }, [selectedTool, tools])

    const selectedToolData = tools.find(t => t.name === selectedTool)

    const validateParameters = () => {
        const errors: Record<string, string> = {}

        if (selectedToolData) {
            Object.entries(selectedToolData.parameters).forEach(([paramName, paramInfo]) => {
                if (paramInfo.required && (!parameters[paramName] || parameters[paramName].toString().trim() === '')) {
                    errors[paramName] = `${paramName} is required`
                }
            })
        }

        setValidationErrors(errors)
        return Object.keys(errors).length === 0
    }

    const handleExecute = () => {
        if (validateParameters()) {
            // Convert parameters to appropriate types
            const processedParams: Record<string, any> = {}

            if (selectedToolData) {
                Object.entries(parameters).forEach(([key, value]) => {
                    const paramInfo = selectedToolData.parameters[key]
                    if (paramInfo) {
                        switch (paramInfo.type) {
                            case 'number':
                                processedParams[key] = value ? Number(value) : undefined
                                break
                            case 'boolean':
                                processedParams[key] = value === 'true' || value === true
                                break
                            default:
                                processedParams[key] = value
                        }
                    }
                })
            }

            onExecute(selectedTool, processedParams)
        }
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleExecute()
        }
        if (e.key === 'Escape') {
            onCancel()
        }
    }

    return (
        <AnimatePresence>
            {isVisible && (
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
                        className="bg-black rounded-lg shadow-2xl backdrop-blur-sm p-6 w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto"
                        style={{
                            background: 'rgba(0, 0, 0, 0.95)',
                            border: '1px solid rgba(97, 253, 252, 0.3)',
                            boxShadow: '0 0 30px rgba(97, 253, 252, 0.3)'
                        }}
                        onKeyDown={handleKeyPress}
                    >
                        <div className="text-center mb-6">
                            <h2 className="text-xl font-mono glow mb-2" style={{ color: '#61FDFC' }}>
                                🛠️ Tool Selection
                            </h2>
                            <p className="text-sm font-mono" style={{ color: '#61FDFC', opacity: 0.8 }}>
                                The AI wants to use a tool. Please configure and confirm.
                            </p>
                        </div>

                        {thinking && (
                            <div className="mb-6">
                                <h3 className="text-sm font-mono mb-2" style={{ color: '#61FDFC' }}>
                                    🧠 AI Thinking:
                                </h3>
                                <div
                                    className="p-3 rounded border text-sm font-mono"
                                    style={{
                                        backgroundColor: 'rgba(97, 253, 252, 0.1)',
                                        borderColor: 'rgba(97, 253, 252, 0.3)',
                                        color: '#61FDFC',
                                        fontStyle: 'italic'
                                    }}
                                >
                                    {thinking}
                                </div>
                            </div>
                        )}

                        <div className="space-y-4">
                            {/* Tool Selection */}
                            <div>
                                <label className="block text-sm font-mono mb-2" style={{ color: '#61FDFC' }}>
                                    Select Tool:
                                </label>
                                <select
                                    value={selectedTool}
                                    onChange={(e) => setSelectedTool(e.target.value)}
                                    className="w-full px-3 py-2 bg-black border rounded font-mono text-sm"
                                    style={{
                                        borderColor: 'rgba(97, 253, 252, 0.3)',
                                        color: '#61FDFC',
                                        backgroundColor: 'rgba(0, 0, 0, 0.8)'
                                    }}
                                >
                                    {tools.map(tool => (
                                        <option key={tool.name} value={tool.name}>
                                            {tool.name} - {tool.description}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            {/* Tool Description */}
                            {selectedToolData && (
                                <div
                                    className="p-3 rounded border text-sm font-mono"
                                    style={{
                                        backgroundColor: 'rgba(97, 253, 252, 0.05)',
                                        borderColor: 'rgba(97, 253, 252, 0.2)',
                                        color: '#61FDFC',
                                        opacity: 0.8
                                    }}
                                >
                                    {selectedToolData.description}
                                </div>
                            )}

                            {/* Parameters */}
                            {selectedToolData && Object.keys(selectedToolData.parameters).length > 0 && (
                                <div>
                                    <h3 className="text-sm font-mono mb-3" style={{ color: '#61FDFC' }}>
                                        Parameters:
                                    </h3>
                                    <div className="space-y-3">
                                        {Object.entries(selectedToolData.parameters).map(([paramName, paramInfo]) => (
                                            <div key={paramName}>
                                                <label className="block text-xs font-mono mb-1" style={{ color: '#61FDFC' }}>
                                                    {paramName}
                                                    {paramInfo.required && <span className="text-red-400 ml-1">*</span>}
                                                    <span className="text-gray-500 ml-2">({paramInfo.type})</span>
                                                </label>
                                                <input
                                                    type={paramInfo.type === 'number' ? 'number' : 'text'}
                                                    value={parameters[paramName] || ''}
                                                    onChange={(e) => setParameters({
                                                        ...parameters,
                                                        [paramName]: e.target.value
                                                    })}
                                                    placeholder={paramInfo.description}
                                                    className="w-full px-3 py-2 bg-black border rounded font-mono text-xs"
                                                    style={{
                                                        borderColor: validationErrors[paramName]
                                                            ? 'rgba(239, 68, 68, 0.5)'
                                                            : 'rgba(97, 253, 252, 0.3)',
                                                        color: '#61FDFC',
                                                        backgroundColor: 'rgba(0, 0, 0, 0.8)'
                                                    }}
                                                />
                                                {validationErrors[paramName] && (
                                                    <div className="text-red-400 text-xs font-mono mt-1">
                                                        {validationErrors[paramName]}
                                                    </div>
                                                )}
                                                <div className="text-xs font-mono mt-1" style={{ color: '#61FDFC', opacity: 0.6 }}>
                                                    {paramInfo.description}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="flex justify-end space-x-3 mt-6">
                            <button
                                onClick={onCancel}
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
                                onClick={handleExecute}
                                disabled={!selectedTool}
                                className="px-4 py-2 text-sm font-mono border rounded transition-colors disabled:opacity-50"
                                style={{
                                    borderColor: 'rgba(97, 253, 252, 0.3)',
                                    color: !selectedTool ? '#61FDFC' : '#000',
                                    backgroundColor: !selectedTool ? 'transparent' : '#61FDFC'
                                }}
                            >
                                Execute Tool
                            </button>
                        </div>

                        <div className="text-xs font-mono mt-3 text-center" style={{ color: '#61FDFC', opacity: 0.6 }}>
                            Press Enter to execute • ESC to cancel
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    )
} 