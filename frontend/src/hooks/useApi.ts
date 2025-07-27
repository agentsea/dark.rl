import { useState, useCallback } from 'react';
import { type SWRResponse } from 'swr';
import { type Message } from './useWebSocket';

const API_BASE_URL = '/v1'; // Use /v1 as the base for all API calls

export interface Task {
    id: string;
    title: string;
    initial_prompt: string;
    model: string;
    mcp_servers: string[];
    created_at: string;
    updated_at: string;
    status: string;
    current_state: string;
}

export interface TaskData {
    task: Task;
    messages: Message[];
}

export interface MCPServer {
    id: string;
    name: string;
    description: string;
    required_env: string[];
    optional_env: string[];
    api_key_available: boolean;
}

export function useApi(taskId: string | undefined, swr: SWRResponse<TaskData, any>) {
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamingResponse, setStreamingResponse] = useState('');

    const { mutate } = swr;

    const streamChatCompletion = useCallback(async (messages: Message[], model: string, adapter?: string) => {
        setIsStreaming(true);
        setStreamingResponse('');

        const response = await fetch(`${API_BASE_URL}/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: model,
                messages: messages,
                stream: true,
                adapter: adapter,
            }),
        });

        if (!response.body) {
            setIsStreaming(false);
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\\n\\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.substring(6);
                    if (data.trim() === '[DONE]') {
                        break;
                    }
                    try {
                        const chunk = JSON.parse(data);
                        const content = chunk.choices[0]?.delta?.content;
                        if (content) {
                            setStreamingResponse(prev => prev + content);
                        }
                    } catch (e) {
                        console.error('Error parsing stream chunk:', e);
                    }
                }
            }
        }

        // Add final response to messages and trigger SWR revalidation
        await mutate((currentData) => {
            if (!currentData) return;
            const finalMessage: Message = { role: 'assistant', content: streamingResponse };
            return {
                ...currentData,
                messages: [...currentData.messages, finalMessage],
            };
        }, false); // optimistic update

        setIsStreaming(false);
        setStreamingResponse('');

    }, [taskId, mutate]);


    const learn = useCallback(async (messages: Message[], adapter: string) => {
        await fetch(`${API_BASE_URL}/learn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages,
                adapter,
            }),
        });
    }, []);

    const createTask = useCallback(async (prompt: string, model: string = 'qwen2.5-vl'): Promise<string | null> => {
        try {
            const response = await fetch(`${API_BASE_URL}/tasks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ initial_prompt: prompt, model }),
            });

            if (!response.ok) {
                console.error('Failed to create task:', response.statusText);
                return null;
            }

            const data = await response.json();
            return data.task_id;
        } catch (error) {
            console.error('Error creating task:', error);
            return null;
        }
    }, []);

    const getMcpServers = useCallback(async (query: string = ''): Promise<MCPServer[]> => {
        try {
            const response = await fetch(`${API_BASE_URL}/mcp/servers?query=${encodeURIComponent(query)}`);
            if (!response.ok) {
                console.error('Failed to fetch MCP servers:', response.statusText);
                return [];
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching MCP servers:', error);
            return [];
        }
    }, []);

    return { isStreaming, streamingResponse, streamChatCompletion, learn, createTask, getMcpServers };
} 