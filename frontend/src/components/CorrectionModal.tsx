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
    const [selectedTool, setSelectedTool] = useState<string>('')
    const [parameters, setParameters] = useState<Record<string, any>>({})
    const [thought, setThought] = useState<string>('')
    const [shouldExecute, setShouldExecute] = useState<boolean>(false)
    const [submitting, setSubmitting] = useState<boolean>(false)

    // Reset state when modal opens
    useEffect(() => {
        if (isOpen) {
            setSelectedTool('')
            setParameters({})
            setThought('')
            setShouldExecute(false)
            setSubmitting(false)
        }
    }, [isOpen])

    // Update parameters when tool selection changes
    useEffect(() => {
        if (selectedTool) {
            const tool = availableTools.find(t => t.name === selectedTool)
            if (tool) {
                const newParams: Record<string, any> = {}
                // Initialize required parameters with empty values
                Object.keys(tool.parameters.properties || {}).forEach(paramName => {
                    const paramInfo = tool.parameters.properties[paramName]
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
        if (!selectedTool || !thought.trim()) {
            alert('Please select a tool and provide your reasoning')
            return
        }

        const tool = availableTools.find(t => t.name === selectedTool)
        if (!tool) return

        // Validate required parameters
        const missingRequired = tool.parameters.required?.filter(paramName => {
            const value = parameters[paramName]
            return value === undefined || value === null || value === ''
        }) || []

        if (missingRequired.length > 0) {
            alert(`Missing required parameters: ${missingRequired.join(', ')}`)
            return
        }

        setSubmitting(true)

        try {
            const result = await onSubmitCorrection(
                currentTaskId,
                messageIndex,
                { name: selectedTool, arguments: parameters },
                thought,
                shouldExecute
            )

            if (result.success) {
                alert('Correction submitted successfully!')
                onClose()
            } else {
                alert(`Error: ${result.error || 'Unknown error'}`)
            }
        } catch (error) {
            alert(`Error submitting correction: ${error}`)
        } finally {
            setSubmitting(false)
        }
    }

    const selectedToolObj = availableTools.find(t => t.name === selectedTool)

    if (!isOpen) return null

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b">
                    <h2 className="text-xl font-semibold text-gray-900">Correct Response</h2>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-gray-600 transition-colors text-2xl"
                    >
                        ×
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 space-y-6">
                    {/* Tool Selection */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Select Tool
                        </label>
                        <select
                            value={selectedTool}
                            onChange={(e) => setSelectedTool(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                            <option value="">-- Select a tool --</option>
                            {availableTools.map(tool => (
                                <option key={tool.name} value={tool.name}>
                                    {tool.name} - {tool.description}
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* Tool Description */}
                    {selectedToolObj && (
                        <div className="bg-blue-50 p-4 rounded-md">
                            <h4 className="font-medium text-blue-900 mb-2">Tool Description</h4>
                            <p className="text-blue-800 text-sm">{selectedToolObj.description}</p>
                        </div>
                    )}

                    {/* Parameters */}
                    {selectedToolObj && Object.keys(selectedToolObj.parameters.properties || {}).length > 0 && (
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-3">
                                Parameters
                            </label>
                            <div className="space-y-4">
                                {Object.entries(selectedToolObj.parameters.properties || {}).map(([paramName, paramInfo]) => {
                                    const isRequired = selectedToolObj.parameters.required?.includes(paramName) || false

                                    return (
                                        <div key={paramName} className="border border-gray-200 rounded-md p-4">
                                            <div className="flex items-center mb-2">
                                                <label className="block text-sm font-medium text-gray-700">
                                                    {paramName}
                                                    {isRequired && <span className="text-red-500 ml-1">*</span>}
                                                </label>
                                                <span className="ml-2 text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                                                    {paramInfo.type}
                                                </span>
                                            </div>
                                            <p className="text-xs text-gray-600 mb-2">{paramInfo.description}</p>

                                            {paramInfo.type === 'boolean' ? (
                                                <input
                                                    type="checkbox"
                                                    checked={parameters[paramName] || false}
                                                    onChange={(e) => handleParameterChange(paramName, e.target.checked)}
                                                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                                />
                                            ) : paramInfo.type === 'number' ? (
                                                <input
                                                    type="number"
                                                    value={parameters[paramName] || ''}
                                                    onChange={(e) => handleParameterChange(paramName, parseFloat(e.target.value) || 0)}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                                                    placeholder={`Enter ${paramName}`}
                                                />
                                            ) : paramInfo.type === 'array' ? (
                                                <textarea
                                                    value={Array.isArray(parameters[paramName]) ? parameters[paramName].join('\n') : ''}
                                                    onChange={(e) => handleParameterChange(paramName, e.target.value.split('\n').filter(v => v.trim()))}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                                                    rows={3}
                                                    placeholder="Enter each value on a new line"
                                                />
                                            ) : (
                                                <input
                                                    type="text"
                                                    value={parameters[paramName] || ''}
                                                    onChange={(e) => handleParameterChange(paramName, e.target.value)}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Your Reasoning <span className="text-red-500">*</span>
                        </label>
                        <textarea
                            value={thought}
                            onChange={(e) => setThought(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            rows={4}
                            placeholder="Explain why this correction is needed and what the tool should do..."
                        />
                    </div>

                    {/* Execute Option */}
                    <div className="flex items-center space-x-3">
                        <input
                            type="checkbox"
                            id="shouldExecute"
                            checked={shouldExecute}
                            onChange={(e) => setShouldExecute(e.target.checked)}
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <label htmlFor="shouldExecute" className="text-sm text-gray-700">
                            Execute this tool immediately after correction
                        </label>
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end space-x-3 p-6 border-t bg-gray-50">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSubmit}
                        disabled={submitting || !selectedTool || !thought.trim()}
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
                    >
                        {submitting ? (
                            <>
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                                <span>Submitting...</span>
                            </>
                        ) : shouldExecute ? (
                            <>
                                <span>▶️</span>
                                <span>Submit & Execute</span>
                            </>
                        ) : (
                            <>
                                <span>💾</span>
                                <span>Submit Correction</span>
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    )
} 