import React, { useState } from 'react';
import type { MCPServer } from '../hooks/useWebSocket';

interface MCPSidebarProps {
    servers: MCPServer[];
    loadingServers: boolean;
    onServerClick: (server: MCPServer) => void;
}

const MCPSidebar: React.FC<MCPSidebarProps> = ({
    servers,
    loadingServers,
    onServerClick,
}) => {
    const [hoveredServerId, setHoveredServerId] = useState<string | null>(null);

    const sidebarStyle: React.CSSProperties = {
        width: '7.5rem',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        backdropFilter: 'blur(8px)',
        padding: '1rem',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        fontFamily: 'monospace',
        fontSize: '0.875rem'
    };

    const headerStyle: React.CSSProperties = {
        fontSize: '1rem',
        fontWeight: 'bold',
        marginTop: '1rem',
        marginBottom: '1.5rem',
        color: '#67e8f9'
    };

    const serverItemBaseStyle: React.CSSProperties = {
        padding: '0.5rem',
        borderRadius: '0.375rem',
        cursor: 'pointer',
        transition: 'background-color 0.2s ease-in-out'
    };

    const serverItemHoverStyle: React.CSSProperties = {
        backgroundColor: 'rgba(55, 65, 81, 0.5)'
    };

    return (
        <div style={sidebarStyle}>
            <h2 style={headerStyle} className="glow">MCP Servers</h2>
            {loadingServers ? (
                <div style={{ color: '#9CA3AF' }}>Loading servers...</div>
            ) : (
                <ul style={{ listStyle: 'none', padding: 0, margin: 0, overflowY: 'auto', flexGrow: 1 }}>
                    {servers.map((server, index) => (
                        <li key={server.id} style={{ marginTop: index > 0 ? '0.5rem' : 0 }}>
                            <div
                                style={{
                                    ...serverItemBaseStyle,
                                    ...(hoveredServerId === server.id ? serverItemHoverStyle : {})
                                }}
                                onClick={() => onServerClick(server)}
                                onMouseEnter={() => setHoveredServerId(server.id)}
                                onMouseLeave={() => setHoveredServerId(null)}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ fontWeight: '600', color: '#E5E7EB' }}>{server.name}</span>
                                </div>
                            </div>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
};

export default MCPSidebar; 