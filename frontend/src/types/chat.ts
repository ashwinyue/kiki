/**
 * Kiki Agent Framework - 聊天相关类型
 */

/**
 * 消息角色
 */
export type MessageRole = 'user' | 'assistant' | 'system';

/**
 * 消息类型
 */
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

/**
 * 会话信息
 */
export interface Session {
  id: string;
  name: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

/**
 * 聊天请求
 */
export interface ChatRequest {
  message: string;
  session_id: string;
  user_id?: string;
}

/**
 * 聊天响应
 */
export interface ChatResponse {
  content: string;
  session_id: string;
}

/**
 * 流式聊天请求
 */
export interface ChatStreamRequest extends ChatRequest {
  stream_mode?: 'messages' | 'updates' | 'values';
}

/**
 * SSE 流事件类型
 */
export type StreamEventType = 'token' | 'update' | 'state' | 'done' | 'error';

/**
 * SSE 流事件
 */
export interface StreamEvent {
  event: StreamEventType;
  data: {
    content?: string;
    session_id: string;
    metadata?: Record<string, unknown>;
  };
}

/**
 * 聊天历史响应
 */
export interface ChatHistoryResponse {
  messages: Message[];
  session_id: string;
}

/**
 * 上下文统计响应
 */
export interface ContextStatsResponse {
  session_id: string;
  message_count: number;
  token_estimate: number;
  role_distribution: {
    user: number;
    assistant: number;
  };
  exists: boolean;
}

/**
 * WebSocket 聊天消息
 */
export interface WSChatMessage {
  action: 'chat';
  prompt: string;
  system?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  session_id?: string;
}

/**
 * WebSocket 聊天事件类型
 */
export type WSChatEventType = 'delta' | 'done' | 'error' | 'metadata' | 'thinking';

/**
 * WebSocket 聊天事件
 */
export interface WSChatEvent {
  type: WSChatEventType;
  content?: string;
  metadata?: Record<string, unknown>;
  timestamp: number;
}
