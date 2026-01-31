/**
 * Kiki Agent Framework - 聊天服务
 */

import api, { request } from './api';
import type {
  ChatRequest,
  ChatResponse,
  ChatStreamRequest,
  ChatHistoryResponse,
  ContextStatsResponse,
} from '@/types/chat';

/**
 * 同步聊天
 */
export async function chat(data: ChatRequest): Promise<ChatResponse> {
  return request(() =>
    api.post<ChatResponse>('/chat', data)
  );
}

/**
 * 获取聊天历史
 */
export async function getChatHistory(sessionId: string): Promise<ChatHistoryResponse> {
  return request(() =>
    api.get<ChatHistoryResponse>(`/chat/history/${sessionId}`)
  );
}

/**
 * 清除聊天历史
 */
export async function clearChatHistory(sessionId: string): Promise<void> {
  return request(() =>
    api.delete(`/chat/history/${sessionId}`)
  );
}

/**
 * 获取上下文统计
 */
export async function getContextStats(sessionId: string): Promise<ContextStatsResponse> {
  return request(() =>
    api.get<ContextStatsResponse>(`/chat/context/${sessionId}/stats`)
  );
}

/**
 * 清除上下文
 */
export async function clearContext(sessionId: string): Promise<void> {
  return request(() =>
    api.delete(`/chat/context/${sessionId}`)
  );
}

/**
 * SSE 流式聊天
 * 返回一个 EventSource 和用于关闭连接的函数
 */
export interface StreamChatOptions {
  onMessage: (data: string) => void;
  onError?: (error: Event) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

export async function streamChat(
  data: ChatStreamRequest,
  options: StreamChatOptions
): Promise<() => void> {
  const { onMessage, onError, onOpen, onClose } = options;

  // 构建 SSE URL
  const baseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1';
  const url = new URL(`${baseUrl}/chat/stream`);

  // 发送 POST 请求获取 SSE
  const response = await fetch(url.toString(), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error('Response body is null');
  }

  onOpen?.();

  // 读取流
  (async () => {
    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          onClose?.();
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data.trim()) {
              try {
                const parsed = JSON.parse(data);
                onMessage(JSON.stringify(parsed));
              } catch {
                onMessage(data);
              }
            }
          }
        }
      }
    } catch (error) {
      onError?.(error as Event);
    }
  })();

  // 返回关闭函数
  return () => {
    reader.cancel();
  };
}

/**
 * WebSocket 聊天
 */
export interface WSChatOptions {
  onMessage: (event: MessageEvent) => void;
  onError?: (event: Event) => void;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
}

export function createWSChatConnection(
  options: WSChatOptions
): WebSocket | null {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = import.meta.env.VITE_API_WS_HOST || window.location.host;
  const baseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1';
  throw new Error('WebSocket streaming has been removed. Use /api/v1/chat/stream (SSE) instead.');

  const ws = new WebSocket(wsUrl);

  ws.onopen = options.onOpen;
  ws.onerror = options.onError;
  ws.onmessage = options.onMessage;
  ws.onclose = options.onClose;

  return ws;
}

export default {
  chat,
  getChatHistory,
  clearChatHistory,
  getContextStats,
  clearContext,
  streamChat,
  createWSChatConnection,
};
