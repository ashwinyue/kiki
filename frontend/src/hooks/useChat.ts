/**
 * Kiki Agent Framework - 聊天 Hook
 */

import { useCallback, useEffect } from 'react';
import { useChatStore } from '@/stores';
import type { Message } from '@/types/chat';

export interface UseChatReturn {
  // 状态
  sessions: Array<{
    id: string;
    name: string;
    message_count: number;
    created_at: string;
    updated_at: string;
  }>;
  currentSessionId: string | null;
  currentMessages: Message[];
  isStreaming: boolean;
  isLoading: boolean;
  error: string | null;

  // 会话操作
  createSession: (name?: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
  switchSession: (sessionId: string) => void;
  loadSessions: () => Promise<void>;

  // 消息操作
  sendMessage: (content: string) => Promise<void>;
  clearMessages: () => Promise<void>;

  // 工具
  clearError: () => void;
}

/**
 * 聊天 Hook
 */
export function useChat(): UseChatReturn {
  const {
    sessions,
    currentSessionId,
    messages,
    isStreaming,
    isLoading,
    error,
    createSession: storeCreateSession,
    deleteSession: storeDeleteSession,
    switchSession: storeSwitchSession,
    loadSessions: storeLoadSessions,
    sendMessage: storeSendMessage,
    clearMessages: storeClearMessages,
    clearError: storeClearError,
  } = useChatStore();

  // 当前会话的消息
  const currentMessages = currentSessionId ? (messages[currentSessionId] || []) : [];

  // 初始化时加载会话列表
  useEffect(() => {
    if (sessions.length === 0) {
      storeLoadSessions().catch(() => {
        // 如果加载失败，创建新会话
        storeCreateSession();
      });
    }
  }, [sessions.length, storeLoadSessions, storeCreateSession]);

  // 创建会话
  const createSession = useCallback(
    async (name?: string) => {
      await storeCreateSession(name);
    },
    [storeCreateSession]
  );

  // 删除会话
  const deleteSession = useCallback(
    async (sessionId: string) => {
      await storeDeleteSession(sessionId);
    },
    [storeDeleteSession]
  );

  // 切换会话
  const switchSession = useCallback(
    (sessionId: string) => {
      storeSwitchSession(sessionId);
    },
    [storeSwitchSession]
  );

  // 加载会话列表
  const loadSessions = useCallback(async () => {
    await storeLoadSessions();
  }, [storeLoadSessions]);

  // 发送消息
  const sendMessage = useCallback(
    async (content: string) => {
      await storeSendMessage(content);
    },
    [storeSendMessage]
  );

  // 清除当前会话的消息
  const clearMessages = useCallback(async () => {
    if (currentSessionId) {
      await storeClearMessages(currentSessionId);
    }
  }, [currentSessionId, storeClearMessages]);

  // 清除错误
  const clearErrorCallback = useCallback(() => {
    storeClearError();
  }, [storeClearError]);

  return {
    sessions,
    currentSessionId,
    currentMessages,
    isStreaming,
    isLoading,
    error,
    createSession,
    deleteSession,
    switchSession,
    loadSessions,
    sendMessage,
    clearMessages,
    clearError: clearErrorCallback,
  };
}

export default useChat;
