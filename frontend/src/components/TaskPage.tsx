import React, { useState, useEffect, useRef, useCallback, type CSSProperties } from 'react';
import { motion } from 'framer-motion';
import { useParams, useNavigate } from 'react-router-dom';
import useSWR from 'swr';
import { useApi, type TaskData, type Task, type Message } from '../hooks/useApi';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import yaml from 'js-yaml';
import TypingAnimation from './TypingAnimation';

interface PendingDualResponse {
    task_id: string
    local_response: string
    gpt_response: string
    local_model: string
    gpt_model: string
    local_finished: boolean
    gpt_finished: boolean
    session_id: number
}

// Component to display tool responses with collapsible JSON
function ToolResponseDisplay({ content }: { content: string }): React.JSX.Element {
    const [isExpanded, setIsExpanded] = useState(false)

    try {
        const jsonData = JSON.parse(content)
        const isSuccess = jsonData.success === true
        const yamlData = yaml.dump(jsonData)

        const customStyle: { [key: string]: CSSProperties } = {
            ...vscDarkPlus,
            'pre[class*="language-"]': {
                ...(vscDarkPlus['pre[class*="language-"]'] as CSSProperties),
                backgroundColor: 'rgba(0, 0, 0, 0.3)',
                border: '1px solid rgba(107, 114, 128, 0.3)',
                borderRadius: '4px',
                padding: '12px',
                fontSize: '12px',
                fontFamily: 'monospace',
                color: '#E5E7EB',
                overflowY: 'auto',
                maxHeight: '400px'
            },
            'code[class*="language-"]': {
                ...(vscDarkPlus['code[class*="language-"]'] as CSSProperties),
                fontFamily: 'monospace',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all' as any,
            },
            property: { ...(vscDarkPlus.property as CSSProperties), color: '#85a6fd' }, // Keys
            string: { ...(vscDarkPlus.string as CSSProperties), color: '#a0e8a7' }, // String values
            number: { ...(vscDarkPlus.number as CSSProperties), color: '#f08d8d' }, // Number values
            boolean: { ...(vscDarkPlus.boolean as CSSProperties), color: '#f08d8d' } // Boolean values
        }

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
                        <SyntaxHighlighter
                            language="yaml"
                            style={customStyle}
                            wrapLines={true}
                            wrapLongLines={true}
                        >
                            {yamlData}
                        </SyntaxHighlighter>
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
    if (!content) return <></>;
    const trimmedContent = content.trim();
    const parts: React.JSX.Element[] = [];
    let currentIndex = 0;

    // Find all <think>, <tool_call>, and <tool_response> tags
    const thinkRegex = /<think>(.*?)<\/think>/gs;
    const toolCallRegex = /<tool_call>(.*?)<\/tool_call>/gs;
    const toolResponseRegex = /<tool_response>(.*?)<\/tool_response>/gs;

    // Combine all patterns and find all matches with their positions
    const allMatches: Array<{ match: RegExpExecArray; type: 'think' | 'tool_call' | 'tool_response' }> = [];

    let match;
    while ((match = thinkRegex.exec(trimmedContent)) !== null) {
        allMatches.push({ match, type: 'think' });
    }

    while ((match = toolCallRegex.exec(trimmedContent)) !== null) {
        allMatches.push({ match, type: 'tool_call' });
    }

    while ((match = toolResponseRegex.exec(trimmedContent)) !== null) {
        allMatches.push({ match, type: 'tool_response' });
    }

    // Sort matches by position
    allMatches.sort((a, b) => a.match.index - b.match.index);

    // Process each match
    allMatches.forEach((matchObj, index) => {
        const { match, type } = matchObj;

        // Add text before this match
        if (match.index > currentIndex) {
            const beforeText = trimmedContent.slice(currentIndex, match.index);
            if (beforeText) {
                parts.push(<span key={`text-${index}`}>{beforeText}</span>);
            }
        }

        // Add the special tag content
        if (type === 'think') {
            parts.push(
                <div key={`think-${index}`} style={{
                    color: '#9CA3AF',
                    fontSize: '0.8rem', // Smaller font size
                    fontStyle: 'italic',
                    marginTop: '8px',
                    marginBottom: '8px',
                    paddingLeft: '16px',
                    borderLeft: '2px solid #4B5563'
                }}>
                    💭 {match[1].trim()}
                </div>
            );
        } else if (type === 'tool_call') {
            try {
                const toolCallData = JSON.parse(match[1].trim())
                const yamlData = yaml.dump(toolCallData)
                const customStyle: { [key: string]: CSSProperties } = {
                    ...vscDarkPlus,
                    'pre[class*="language-"]': {
                        ...(vscDarkPlus['pre[class*="language-"]'] as CSSProperties),
                        color: '#E5E7EB',
                        fontSize: '0.85em',
                        fontFamily: 'monospace',
                        margin: 0,
                        padding: 0,
                        background: 'none',
                    },
                    'code[class*="language-"]': {
                        ...(vscDarkPlus['code[class*="language-"]'] as CSSProperties),
                        fontFamily: 'monospace',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-all' as any,
                    },
                    property: { ...(vscDarkPlus.property as CSSProperties), color: '#85a6fd' }, // Keys
                    string: { ...(vscDarkPlus.string as CSSProperties), color: '#a0e8a7' }, // String values
                    number: { ...(vscDarkPlus.number as CSSProperties), color: '#f08d8d' }, // Number values
                    boolean: { ...(vscDarkPlus.boolean as CSSProperties), color: '#f08d8d' } // Boolean values
                }
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
                        <SyntaxHighlighter
                            language="yaml"
                            style={customStyle}
                            wrapLines={true}
                            wrapLongLines={true}
                        >
                            {yamlData}
                        </SyntaxHighlighter>
                    </div>
                );
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
                );
            }
        } else if (type === 'tool_response') {
            // Use the same ToolResponseDisplay component for tool responses
            parts.push(
                <div key={`tool-response-${index}`} style={{ marginTop: '8px', marginBottom: '8px' }}>
                    <ToolResponseDisplay content={match[1]} />
                </div>
            );
        }

        currentIndex = match.index + match[0].length;
    });

    // Add any remaining text after the last match
    if (currentIndex < trimmedContent.length) {
        const remainingText = trimmedContent.slice(currentIndex);
        if (remainingText) {
            parts.push(<span key="remaining">{remainingText}</span>);
        }
    }

    // If no special tags found, return the original content
    if (parts.length === 0) {
        return <span>{trimmedContent}</span>;
    }

    return <>{parts}</>;
}

function TaskPage() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const [userInput, setUserInput] = useState('');
    const bottomRef = useRef<HTMLDivElement>(null);

    const fetcher = (url: string) => fetch(url).then(r => r.json());

    const swrResponse = useSWR<TaskData>(id ? `/api/task/${id}` : null, fetcher, {
        refreshInterval: 5000,
    });

    const { data: taskData, error: taskError, mutate } = swrResponse;
    const { isStreaming, streamingResponse, streamChatCompletion, learn } = useApi(id, swrResponse);

    const task = taskData?.task;
    const taskMessages = taskData?.messages || [];

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!userInput.trim() || !task || !taskData) return;

        const newUserMessage: Message = { role: 'user', content: userInput };

        // Optimistic UI update
        mutate({ ...taskData, messages: [...taskMessages, newUserMessage] }, false);
        setUserInput('');

        await streamChatCompletion([...taskMessages, newUserMessage], task.model);
    };

    useEffect(() => {
        setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
    }, [taskMessages, streamingResponse]);

    if (taskError) {
        return (
            <div className="min-h-screen flex items-center justify-center p-8">
                <div className="text-xl font-mono glow text-red-400">
                    Failed to load task: {taskError.message}
                </div>
            </div>
        );
    }

    if (!taskData) {
        return (
            <div className="min-h-screen flex items-center justify-center p-8">
                <div className="text-xl font-mono glow">
                    Loading task...
                </div>
            </div>
        );
    }

    return (
        <div className="flex min-h-screen bg-black text-gray-200">
            <div className="flex-1 relative flex flex-col">
                <div className="flex-grow p-8 pt-20">
                    <div className="max-w-4xl mx-auto w-full">
                        <motion.div
                            className="text-center mb-12"
                            initial={{ opacity: 0, y: -20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            <h1 className="text-xl font-mono glow">{task?.title}</h1>
                            <p className="text-sm font-mono opacity-60 mt-2">
                                Model: {task?.model} | Task ID: {task?.id}
                            </p>
                        </motion.div>

                        <div>
                            {taskMessages.map((message, index) => (
                                <div key={index} className="mb-10">
                                    <div className="mb-2 text-cyan-300 glow">
                                        {message.role.charAt(0).toUpperCase() + message.role.slice(1)}
                                    </div>
                                    <div className="border-l-2 border-cyan-500 pl-5 ml-2 text-sm">
                                        {/* Using a placeholder for parseContent */}
                                        <div>{message.content}</div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {isStreaming && (
                            <div className="mb-10">
                                <div className="mb-2 text-cyan-300 glow">Assistant</div>
                                <div className="border-l-2 border-cyan-500 pl-5 ml-2 text-sm">
                                    {/* Using a placeholder for parseContent */}
                                    <div>{streamingResponse}</div>
                                    <span className="animate-pulse ml-2">▋</span>
                                </div>
                            </div>
                        )}

                        <div ref={bottomRef} />
                    </div>
                </div>

                <div className="sticky bottom-0 left-0 right-0 bg-black/50 backdrop-blur-sm p-4">
                    <form onSubmit={handleSubmit} className="mx-auto max-w-3xl">
                        <div className="relative">
                            <textarea
                                value={userInput}
                                onChange={(e) => setUserInput(e.target.value)}
                                placeholder="Send a follow-up message..."
                                className="form-textarea w-full min-h-[80px] text-lg font-mono resize-none glow"
                                onKeyPress={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        (e.target as HTMLFormElement).requestSubmit();
                                    }
                                }}
                            />
                            <div className="absolute bottom-3 right-3">
                                <button type="submit" disabled={isStreaming} className="text-xs font-mono glow border rounded px-4 py-2 disabled:opacity-50">
                                    Send
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    );
}

export default TaskPage;

