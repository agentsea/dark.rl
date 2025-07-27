import React from 'react';
import type { MCPServer, MCPServerAction } from '../hooks/useWebSocket';
import { motion, AnimatePresence } from 'framer-motion';
import yaml from 'js-yaml'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface MCPToolsModalProps {
    isOpen: boolean;
    onClose: () => void;
    server: MCPServer | null;
    actions: MCPServerAction[];
    loading: boolean;
}

const MCPToolsModal: React.FC<MCPToolsModalProps> = ({ isOpen, onClose, server, actions, loading }) => {
    if (!server) return null;

    const customStyle = {
        ...vscDarkPlus,
        'pre[class*="language-"]': {
            ...(vscDarkPlus['pre[class*="language-"]'] as React.CSSProperties),
            backgroundColor: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(107, 114, 128, 0.3)',
            borderRadius: '4px',
            padding: '12px',
            fontSize: '12px',
            fontFamily: 'monospace',
            color: '#E5E7EB',
            overflowY: 'auto',
            maxHeight: '200px'
        },
    };

    const backdropStyle: React.CSSProperties = {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        backdropFilter: 'blur(4px)',
        zIndex: 50,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    };

    const modalStyle: React.CSSProperties = {
        backgroundColor: '#000',
        border: '1px solid rgba(56, 189, 248, 0.3)',
        borderRadius: '0.5rem',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
        width: '100%',
        maxWidth: '42rem',
        maxHeight: '80vh',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'monospace',
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    style={backdropStyle}
                    onClick={onClose}
                >
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.9, opacity: 0 }}
                        style={modalStyle}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div style={{ padding: '1rem', borderBottom: '1px solid rgba(55, 65, 81, 0.5)' }}>
                            <h2 className="text-xl font-bold text-cyan-300 glow">{server.name} Tools</h2>
                            <p className="text-sm text-gray-400">{server.description}</p>
                        </div>
                        <div style={{ padding: '1rem', overflowY: 'auto' }}>
                            {loading ? (
                                <div className="text-gray-400">Loading tools...</div>
                            ) : (
                                <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                    {actions.map((action) => (
                                        <li key={action.name} style={{ padding: '1rem', borderRadius: '0.375rem', backgroundColor: 'rgba(31, 41, 55, 0.5)', border: '1px solid rgba(55, 65, 81, 0.5)' }}>
                                            <h3 className="font-bold text-cyan-400">{action.name}</h3>
                                            <p className="text-gray-300 text-sm mt-1">{action.description}</p>
                                            {action.parameters && Object.keys(action.parameters.properties).length > 0 && (
                                                <div className="mt-3">
                                                    <h4 className="text-sm font-semibold text-gray-400 mb-1">Parameters:</h4>
                                                    <SyntaxHighlighter
                                                        language="yaml"
                                                        style={customStyle as any}
                                                        wrapLines={true}
                                                        wrapLongLines={true}
                                                    >
                                                        {yaml.dump(action.parameters)}
                                                    </SyntaxHighlighter>
                                                </div>
                                            )}
                                        </li>
                                    ))}
                                    {actions.length === 0 && (
                                        <div className="text-gray-500 text-center p-4">No tools found for this server.</div>
                                    )}
                                </ul>
                            )}
                        </div>
                        <div style={{ padding: '1rem', borderTop: '1px solid rgba(55, 65, 81, 0.5)', textAlign: 'right' }}>
                            <button
                                onClick={onClose}
                                className="px-4 py-2 border rounded font-mono backdrop-blur text-sm"
                                style={{
                                    backgroundColor: 'rgba(0, 0, 0, 0.4)',
                                    borderColor: 'rgba(107, 114, 128, 0.3)',
                                    color: '#61FDFC'
                                }}
                            >
                                Close
                            </button>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default MCPToolsModal; 