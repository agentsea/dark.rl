import React, { useState, useEffect } from 'react'

interface Tool {
    name: string
    description: string
    parameters: {
        type: string
        properties: Record<string, {
            type: string
            description: string
            required?: boolean
        }>
        required: string[]
    }
}

interface CorrectionModalProps {
    isOpen: boolean
    onClose: () => void
    availableTools: Tool[]
    currentTaskId: string
    messageIndex: number
    onSubmitCorrection: (
        taskId: string,
        messageIndex: number,
        correctedToolCall: { name: string; arguments: Record<string, any> },
        thought: string,
        shouldExecute: boolean
    ) => Promise<any>
}

export default function CorrectionModal({
    isOpen,
    onClose,
    availableTools,
    currentTaskId,
    messageIndex,
    onSubmitCorrection
}: CorrectionModalProps) {
    console.log('🔧 CorrectionModal render - isOpen:', isOpen, 'taskId:', currentTaskId, 'messageIndex:', messageIndex)
    // Remove the repetitive logs about available tools count that show on every render

    const [selectedTool, setSelectedTool] = useState<string>('')
    const [parameters, setParameters] = useState<Record<string, any>>({})
    const [thought, setThought] = useState<string>('')
    const [submitting, setSubmitting] = useState<boolean>(false)

    // Reset state when modal opens
    useEffect(() => {
        if (isOpen) {
            setSelectedTool('')
            setParameters({})
            setThought('')
            setSubmitting(false)
        }
    }, [isOpen])

    // Update parameters when tool selection changes
    useEffect(() => {
        if (selectedTool) {
            console.log('🔧 Tool selected:', selectedTool)
            const tool = availableTools.find(t => t.name === selectedTool)
            console.log('🛠️ Found tool:', tool)
            if (tool) {
                console.log('📝 Tool parameters:', tool.parameters)
                console.log('📝 Tool parameters properties:', tool.parameters.properties)
                const newParams: Record<string, any> = {}
                // Initialize required parameters with empty values
                Object.keys(tool.parameters.properties || {}).forEach(paramName => {
                    const paramInfo = tool.parameters.properties[paramName]
                    console.log(`🔧 Processing parameter ${paramName}:`, paramInfo)
                    if (paramInfo.type === 'boolean') {
                        newParams[paramName] = false
                    } else if (paramInfo.type === 'number') {
                        newParams[paramName] = 0
                    } else if (paramInfo.type === 'array') {
                        newParams[paramName] = []
                    } else {
                        newParams[paramName] = ''
                    }
                })
                console.log('🔧 Setting parameters:', newParams)
                setParameters(newParams)
            }
        }
    }, [selectedTool, availableTools])

    const handleParameterChange = (paramName: string, value: any) => {
        setParameters(prev => ({
            ...prev,
            [paramName]: value
        }))
    }

    const handleSubmit = async () => {
        console.log('🔧 Correct button clicked - selectedTool:', selectedTool, 'thought:', thought)

        if (!selectedTool || !thought.trim()) {
            console.log('❌ Validation failed - missing tool or thought')
            alert('Please select a tool and provide your reasoning')
            return
        }

        const tool = availableTools.find(t => t.name === selectedTool)
        if (!tool) {
            console.log('❌ Tool not found:', selectedTool)
            return
        }

        // Validate required parameters
        const missingRequired = tool.parameters.required?.filter(paramName => {
            const value = parameters[paramName]
            return value === undefined || value === null || value === ''
        }) || []

        if (missingRequired.length > 0) {
            console.log('❌ Missing required parameters:', missingRequired)
            alert(`Missing required parameters: ${missingRequired.join(', ')}`)
            return
        }

        console.log('📝 Submitting correction with:', {
            taskId: currentTaskId,
            messageIndex,
            toolCall: { name: selectedTool, arguments: parameters },
            thought,
            shouldExecute: true
        })

        setSubmitting(true)

        try {
            const result = await onSubmitCorrection(
                currentTaskId,
                messageIndex,
                { name: selectedTool, arguments: parameters },
                thought,
                true // Always execute after correction
            )

            console.log('🔄 onSubmitCorrection result:', result)

            if (result.success) {
                console.log('✅ Correction submitted successfully!')
                onClose()
            } else {
                console.log('❌ Error submitting correction:', result.error)
                alert(`Error: ${result.error || 'Unknown error'}`)
            }
        } catch (error) {
            console.log('❌ Exception submitting correction:', error)
            alert(`Error submitting correction: ${error}`)
        } finally {
            setSubmitting(false)
        }
    }

    const selectedToolObj = availableTools.find(t => t.name === selectedTool)

    // Debug selected tool and parameters
    if (selectedToolObj) {
        console.log('🔧 Selected tool object:', selectedToolObj)
        console.log('🔧 Tool parameters:', selectedToolObj.parameters)
        console.log('🔧 Tool parameters properties:', selectedToolObj.parameters?.properties)
        console.log('🔧 Parameters properties keys:', Object.keys(selectedToolObj.parameters?.properties || {}))
        console.log('🔧 Should show parameters section?', Object.keys(selectedToolObj.parameters?.properties || {}).length > 0)
    }

    if (!isOpen) {
        // Remove noisy log that shows on every render when modal is closed
        return null
    }

    console.log('✅ CorrectionModal rendering!')

    return (
        <div
            className="fixed inset-0 flex items-center justify-center z-50"
            style={{
                zIndex: 9999,
                backgroundColor: 'rgba(0, 0, 0, 0.7)', // Dark grey overlay
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0
            }}
        >
            <div
                className="rounded-lg shadow-xl mx-4 max-h-[80vh] overflow-y-auto"
                style={{
                    backgroundColor: 'rgba(0, 0, 0, 0.9)', // Dark background
                    border: '1px solid rgba(107, 114, 128, 0.3)', // Subtle grey border
                    zIndex: 10000,
                    width: '500px', // Fixed width instead of responsive
                    maxWidth: '90vw' // Don't exceed viewport width on small screens
                }}
            >
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b" style={{ borderColor: 'rgba(107, 114, 128, 0.3)' }}>
                    <h2 className="text-xl font-semibold font-mono" style={{ color: '#61FDFC' }}>Correct Response</h2>
                    <button
                        onClick={onClose}
                        className="transition-colors text-2xl hover:scale-105 w-8 h-8 flex items-center justify-center rounded"
                        style={{
                            color: '#9CA3AF',
                            backgroundColor: 'rgba(0, 0, 0, 0.3)',
                            border: '1px solid rgba(107, 114, 128, 0.3)'
                        }}
                    >
                        ×
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 space-y-6">
                    {/* Tool Selection */}
                    <div>
                        <label className="block text-sm font-medium font-mono mb-2" style={{ color: '#61FDFC' }}>
                            Select Tool
                        </label>
                        <select
                            value={selectedTool}
                            onChange={(e) => setSelectedTool(e.target.value)}
                            className="w-full px-3 py-2 border rounded-md focus:outline-none font-mono text-sm"
                            style={{
                                backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                borderColor: 'rgba(107, 114, 128, 0.3)',
                                color: '#9CA3AF'
                            }}
                        >
                            <option value="" style={{ backgroundColor: 'rgba(0, 0, 0, 0.9)', color: '#9CA3AF' }}>-- Select a tool --</option>
                            {availableTools.map(tool => (
                                <option key={tool.name} value={tool.name} style={{ backgroundColor: 'rgba(0, 0, 0, 0.9)', color: '#9CA3AF' }}>
                                    {tool.name} - {tool.description}
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* Tool Description */}
                    {selectedToolObj && (
                        <div className="p-4 rounded-md" style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)', border: '1px solid rgba(107, 114, 128, 0.3)' }}>
                            <h4 className="font-medium font-mono mb-2" style={{ color: '#61FDFC' }}>Tool Description</h4>
                            <p className="text-sm font-mono" style={{ color: '#9CA3AF' }}>{selectedToolObj.description}</p>
                        </div>
                    )}

                    {/* Parameters */}
                    {selectedToolObj && Object.keys(selectedToolObj.parameters.properties || {}).length > 0 && (
                        <div>
                            <label className="block text-sm font-medium font-mono mb-3" style={{ color: '#61FDFC' }}>
                                Parameters
                            </label>
                            <div className="space-y-4">
                                {Object.entries(selectedToolObj.parameters.properties || {}).map(([paramName, paramInfo]) => {
                                    const isRequired = selectedToolObj.parameters.required?.includes(paramName) || false

                                    return (
                                        <div key={paramName} className="border rounded-md p-4" style={{ borderColor: 'rgba(107, 114, 128, 0.3)', backgroundColor: 'rgba(0, 0, 0, 0.2)' }}>
                                            <div className="flex items-center mb-2">
                                                <label className="block text-sm font-medium font-mono" style={{ color: '#61FDFC' }}>
                                                    {paramName}
                                                    {isRequired && <span className="text-red-500 ml-1">*</span>}
                                                </label>
                                                <span className="ml-2 text-xs font-mono px-2 py-1 rounded" style={{ color: '#9CA3AF', backgroundColor: 'rgba(0, 0, 0, 0.3)', border: '1px solid rgba(107, 114, 128, 0.3)' }}>
                                                    {paramInfo.type}
                                                </span>
                                            </div>
                                            <p className="text-xs font-mono mb-2" style={{ color: '#9CA3AF' }}>{paramInfo.description}</p>

                                            {paramInfo.type === 'boolean' ? (
                                                <input
                                                    type="checkbox"
                                                    checked={parameters[paramName] || false}
                                                    onChange={(e) => handleParameterChange(paramName, e.target.checked)}
                                                    className="h-4 w-4 rounded"
                                                    style={{ accentColor: '#61FDFC' }}
                                                />
                                            ) : paramInfo.type === 'number' ? (
                                                <input
                                                    type="number"
                                                    value={parameters[paramName] || ''}
                                                    onChange={(e) => handleParameterChange(paramName, parseFloat(e.target.value) || 0)}
                                                    className="w-full px-3 py-2 border rounded-md focus:outline-none font-mono text-sm"
                                                    style={{
                                                        backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                                        borderColor: 'rgba(107, 114, 128, 0.3)',
                                                        color: '#9CA3AF'
                                                    }}
                                                    placeholder={`Enter ${paramName}`}
                                                />
                                            ) : paramInfo.type === 'array' ? (
                                                <textarea
                                                    value={Array.isArray(parameters[paramName]) ? parameters[paramName].join('\n') : ''}
                                                    onChange={(e) => handleParameterChange(paramName, e.target.value.split('\n').filter(v => v.trim()))}
                                                    className="w-full px-3 py-2 border rounded-md focus:outline-none font-mono text-sm"
                                                    style={{
                                                        backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                                        borderColor: 'rgba(107, 114, 128, 0.3)',
                                                        color: '#9CA3AF'
                                                    }}
                                                    rows={3}
                                                    placeholder="Enter each value on a new line"
                                                />
                                            ) : (
                                                <input
                                                    type="text"
                                                    value={parameters[paramName] || ''}
                                                    onChange={(e) => handleParameterChange(paramName, e.target.value)}
                                                    className="w-full px-3 py-2 border rounded-md focus:outline-none font-mono text-sm"
                                                    style={{
                                                        backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                                        borderColor: 'rgba(107, 114, 128, 0.3)',
                                                        color: '#9CA3AF'
                                                    }}
                                                    placeholder={`Enter ${paramName}`}
                                                />
                                            )}
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    )}

                    {/* Thought/Reasoning */}
                    <div>
                        <label className="block text-sm font-medium font-mono mb-2" style={{ color: '#61FDFC' }}>
                            Your Reasoning <span className="text-red-500">*</span>
                        </label>
                        <textarea
                            value={thought}
                            onChange={(e) => setThought(e.target.value)}
                            className="w-full px-3 py-2 border rounded-md focus:outline-none font-mono text-sm"
                            style={{
                                backgroundColor: 'rgba(0, 0, 0, 0.3)',
                                borderColor: 'rgba(107, 114, 128, 0.3)',
                                color: '#9CA3AF'
                            }}
                            rows={4}
                            placeholder="Explain why this correction is needed and what the tool should do..."
                        />
                    </div>


                </div>

                {/* Footer */}
                <div className="flex items-center justify-end space-x-3 p-6 border-t" style={{ borderColor: 'rgba(107, 114, 128, 0.3)', backgroundColor: 'rgba(0, 0, 0, 0.2)' }}>
                    <button
                        onClick={onClose}
                        className="px-4 py-2 border rounded-md transition-colors font-mono text-sm"
                        style={{
                            backgroundColor: 'rgba(0, 0, 0, 0.3)',
                            borderColor: 'rgba(107, 114, 128, 0.3)',
                            color: '#9CA3AF'
                        }}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSubmit}
                        disabled={submitting || !selectedTool || !thought.trim()}
                        className="px-4 py-2 rounded-md transition-colors flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed font-mono text-sm"
                        style={{
                            backgroundColor: '#61FDFC',
                            color: '#000000',
                            border: 'none'
                        }}
                    >
                        {submitting ? (
                            <>
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                                <span>Submitting...</span>
                            </>
                        ) : (
                            <>
                                <span>▶️</span>
                                <span>Submit & Execute</span>
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    )
} 