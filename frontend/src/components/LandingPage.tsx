import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import TypingAnimation from './TypingAnimation';
import MCPServerDropdown from './MCPServerDropdown';
import { APIKeyModal } from './APIKeyModal';
import { useApi, type MCPServer } from '../hooks/useApi';
import useSWR from 'swr';

const ConnectionStatus = {
    CONNECTED: 'CONNECTED',
    CONNECTING: 'CONNECTING',
    DISCONNECTED: 'DISCONNECTED',
    ERROR: 'ERROR',
};

function LandingPage() {
    const [prompt, setPrompt] = useState('');
    const [showMCPDropdown, setShowMCPDropdown] = useState(false);
    const [mcpQuery, setMcpQuery] = useState('');
    const [cursorPosition, setCursorPosition] = useState(0);
    const [visibleExamples, setVisibleExamples] = useState<(number | null)[]>(Array(4).fill(null));
    const [hasInitiallyLoaded, setHasInitiallyLoaded] = useState(false);
    const [pulseTriggers, setPulseTriggers] = useState<number[]>(Array(4).fill(0));
    const [apiKeyServer, setApiKeyServer] = useState<MCPServer | null>(null);
    const [connectionStatus, setConnectionStatus] = useState(ConnectionStatus.CONNECTING);
    const [mcpServers, setMcpServers] = useState<MCPServer[]>([]);
    const [loadingMcpServers, setLoadingMcpServers] = useState(false);

    const navigate = useNavigate();
    const swrResponse = useSWR(null, () => Promise.resolve(null));
    const { createTask, getMcpServers: fetchMcpServers } = useApi(undefined, swrResponse as any);

    const examples = useMemo(() => [
        "Use @playwright to find the best restaurants in Boulder and put them in a document",
        "Help me research quantum computing trends and summarize them",
        "Use @playwright to take a screenshot of the dark.rl website",
        "Plan a trip to Japan using flight-search and hotel-booking tools",
        "Analyze my github repositories and create a portfolio website",
        "Use a weather API to check the forecast and send reminder emails",
        "Search arxiv for ML papers and create a reading list",
        "Use a calendar tool to schedule meetings and send invitations"
    ], []);

    useEffect(() => {
        const connect = async () => {
            try {
                // Simulate a connection attempt
                setConnectionStatus(ConnectionStatus.CONNECTING);
                await new Promise(resolve => setTimeout(resolve, 1000));
                setConnectionStatus(ConnectionStatus.CONNECTED);
            } catch (error) {
                setConnectionStatus(ConnectionStatus.ERROR);
            }
        };
        connect();
    }, []);

    const getMcpServers = async (query: string) => {
        setLoadingMcpServers(true);
        const servers = await fetchMcpServers(query);
        setMcpServers(servers);
        setLoadingMcpServers(false);
    };

    // Initial loading sequence
    useEffect(() => {
        if (connectionStatus !== ConnectionStatus.CONNECTED || hasInitiallyLoaded) return;

        const initialLoadTimeout = setTimeout(() => {
            const availableIndices = examples.map((_, index) => index);
            const selectedIndices: number[] = [];
            for (let i = 0; i < 4; i++) {
                const randomIndex = Math.floor(Math.random() * availableIndices.length);
                selectedIndices.push(availableIndices.splice(randomIndex, 1)[0]);
            }
            setVisibleExamples(selectedIndices);
            setHasInitiallyLoaded(true);
        }, 1000);

        return () => clearTimeout(initialLoadTimeout);
    }, [connectionStatus, hasInitiallyLoaded, examples]);

    // Individual pulse animations for examples
    useEffect(() => {
        if (connectionStatus !== ConnectionStatus.CONNECTED || !hasInitiallyLoaded) return;

        const scheduleRandomPulse = () => {
            const delay = Math.random() * 6000 + 2000;
            setTimeout(() => {
                const availableIndexes = visibleExamples.map((_, index) => index).filter(i => visibleExamples[i] !== null);
                if (availableIndexes.length > 0) {
                    const randomIndex = availableIndexes[Math.floor(Math.random() * availableIndexes.length)];
                    setPulseTriggers(prev => {
                        const newTriggers = [...prev];
                        newTriggers[randomIndex] = Date.now();
                        return newTriggers;
                    });
                }
                scheduleRandomPulse();
            }, delay);
        };
        scheduleRandomPulse();
    }, [connectionStatus, hasInitiallyLoaded, visibleExamples]);

    const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const value = e.target.value;
        const cursor = e.target.selectionStart;
        setPrompt(value);
        setCursorPosition(cursor);
        const atIndex = value.lastIndexOf('@', cursor - 1);
        if (atIndex !== -1) {
            const charBefore = atIndex === 0 ? ' ' : value[atIndex - 1];
            if (charBefore === ' ' || charBefore === '\\n' || atIndex === 0) {
                const afterAt = value.substring(atIndex + 1, cursor);
                if (!afterAt.includes(' ') && !afterAt.includes('\\n')) {
                    setMcpQuery(afterAt);
                    setShowMCPDropdown(true);
                    getMcpServers(afterAt);
                    return;
                }
            }
        }
        setShowMCPDropdown(false);
    };

    const handleMCPServerSelect = (server: MCPServer) => {
        if (!server.api_key_available) {
            setApiKeyServer(server);
            return;
        }
        const atIndex = prompt.lastIndexOf('@', cursorPosition - 1);
        if (atIndex !== -1) {
            const beforeAt = prompt.substring(0, atIndex);
            const afterCursor = prompt.substring(cursorPosition);
            const newPrompt = `${beforeAt}@${server.id} ${afterCursor}`;
            setPrompt(newPrompt);
        }
        setShowMCPDropdown(false);
    };

    const handleApiKeyRequired = (server: MCPServer) => {
        setApiKeyServer(server);
        setShowMCPDropdown(false);
    };

    const handleApiKeyModalSave = async () => {
        getMcpServers(mcpQuery);
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (!prompt.trim() || connectionStatus !== ConnectionStatus.CONNECTED) return;
        const taskId = await createTask(prompt.trim());
        if (taskId) {
            navigate(`/tasks/${taskId}`);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const form = e.currentTarget.closest('form');
            if (form) form.requestSubmit();
        }
    };

    const getStatusColor = () => {
        switch (connectionStatus) {
            case ConnectionStatus.CONNECTED: return 'status-online';
            case ConnectionStatus.CONNECTING: return 'status-connecting';
            default: return 'status-offline';
        }
    };

    const getStatusText = () => connectionStatus;

    return (
        <div className="min-h-screen relative">
            <motion.div className="absolute z-10" style={{ top: '24px', right: '24px' }} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5 }}>
                <div className="backdrop-blur border flex items-center" style={{ backgroundColor: 'rgba(0, 0, 0, 0.4)', borderColor: 'rgba(107, 114, 128, 0.3)', padding: '8px 16px', gap: '12px', borderRadius: '16px' }}>
                    <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full glow`} style={{ backgroundColor: '#61FDFC' }}></div>
                        <span className={`text-xs font-mono ${getStatusColor()}`}>{getStatusText()}</span>
                    </div>
                </div>
            </motion.div>

            {connectionStatus === ConnectionStatus.CONNECTED ? (
                <div className="min-h-screen flex flex-col p-8">
                    <motion.div className="flex justify-center pt-16 pb-8" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
                        <img src="https://storage.googleapis.com/guisurfer-assets/dark_rl_extreme_glow.png" alt="DARK.RL" className="h-[300px] w-[300px] object-contain" />
                    </motion.div>

                    <div className="flex-1 flex justify-center" style={{ alignItems: 'flex-start', paddingTop: '80px' }}>
                        <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.5 }}>
                                <TypingAnimation text="What would you like me to learn today?" speed={40} className="text-2xl glow font-mono" onComplete={() => { }} />
                            </motion.div>
                            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 1.2 }} className="mx-auto" style={{ marginTop: '40px', width: '800px', maxWidth: '90vw', marginLeft: 'auto', marginRight: 'auto' }}>
                                <form onSubmit={handleSubmit} className="space-y-4" style={{ width: '100%' }}>
                                    <div className="relative">
                                        <textarea value={prompt} onChange={handleTextareaChange} onKeyPress={handleKeyPress} className="form-textarea min-h-[120px] w-full glow resize-none font-mono text-lg" placeholder="Enter your learning objective... (type @ to mention MCP servers)" />
                                        <div className="absolute bottom-3 right-3 text-xs text-gray-600">Press Enter to send</div>
                                        <MCPServerDropdown isVisible={showMCPDropdown} servers={mcpServers} loading={loadingMcpServers} onSelect={handleMCPServerSelect} onClose={() => setShowMCPDropdown(false)} query={mcpQuery} onApiKeyRequired={handleApiKeyRequired} />
                                        <APIKeyModal isVisible={!!apiKeyServer} server={apiKeyServer} onClose={() => setApiKeyServer(null)} onSave={handleApiKeyModalSave} />
                                    </div>
                                </form>
                                <div style={{ marginTop: '100px' }}>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gridTemplateRows: '80px', gap: '20px' }}>
                                        {Array.from({ length: 4 }, (_, cellIndex) => {
                                            const assignedExampleIndex = visibleExamples[cellIndex];
                                            const hasExample = assignedExampleIndex !== null && assignedExampleIndex !== undefined;
                                            return (
                                                <div key={cellIndex} className="relative">
                                                    {hasExample && (<motion.div key={`example-${cellIndex}`} className="text-xs font-mono cursor-pointer w-full h-full" initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 2.5, ease: "easeInOut" }} onClick={() => setPrompt(examples[assignedExampleIndex])} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                                                        <motion.div className="rounded-full border backdrop-blur-sm text-center w-full h-full" animate={{ borderColor: pulseTriggers[cellIndex] > 0 ? ['rgba(97, 253, 252, 0.3)', 'rgba(97, 253, 252, 0.6)', 'rgba(97, 253, 252, 0.3)'] : 'rgba(97, 253, 252, 0.3)', backgroundColor: pulseTriggers[cellIndex] > 0 ? ['rgba(97, 253, 252, 0.1)', 'rgba(97, 253, 252, 0.2)', 'rgba(97, 253, 252, 0.1)'] : 'rgba(97, 253, 252, 0.1)', boxShadow: pulseTriggers[cellIndex] > 0 ? ['0 0 10px rgba(97, 253, 252, 0.2)', '0 0 20px rgba(97, 253, 252, 0.4)', '0 0 10px rgba(97, 253, 252, 0.2)'] : '0 0 10px rgba(97, 253, 252, 0.2)' }} transition={{ duration: 1.2, ease: "easeInOut" }} style={{ color: '#61FDFC', display: 'flex', alignItems: 'center', justifyContent: 'center', lineHeight: '1.3', padding: '12px' }}>
                                                            "{examples[assignedExampleIndex]}"
                                                        </motion.div>
                                                    </motion.div>)}
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
                <div className="min-h-screen flex items-center justify-center p-8">
                    <motion.div className="text-center" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
                        <div className="text-gray-600 glow font-mono">
                            {connectionStatus === ConnectionStatus.CONNECTING ? "Establishing neural connection..." : "Neural connection required"}
                        </div>
                    </motion.div>
                </div>
            )}
        </div>
    );
}

export default LandingPage; 