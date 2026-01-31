/**
 * Kiki Agent Framework - 聊天状态管理
 *
 * 使用 Zustand 进行轻量级状态管理
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { Message, Session } from '@/types/chat';
import { chat as chatService, auth as authService } from '@/services';

/**
 * 聊天状态接口
 */
interface ChatState {
  // 状态
  sessions: Session[];
  currentSessionId: string | null;
  messages: Record<string, Message[]>; // sessionId -> messages
  isStreaming: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions - 会话管理
  createSession: (name?: string) => Promise<Session>;
  deleteSession: (sessionId: string) => Promise<void>;
  switchSession: (sessionId: string) => void;
  loadSessions: () => Promise<void>;

  // Actions - 消息管理
  sendMessage: (content: string) => Promise<void>;
  addMessage: (sessionId: string, message: Message) => void;
  updateLastMessage: (sessionId: string, content: string) => void;
  loadMessages: (sessionId: string) => Promise<void>;
  clearMessages: (sessionId: string) => Promise<void>;

  // Actions - 流式消息
  startStreaming: () => void;
  stopStreaming: () => void;

  // Actions - 工具
  clearError: () => void;
}

/**
 * 聊天 Store
 */
export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      // 初始状态
      sessions: [],
      currentSessionId: null,
      messages: {},
      isStreaming: false,
      isLoading: false,
      error: null,

      // 创建会话
      createSession: async (name = '新对话') => {
        try {
          const response = await authService.createSession({ name });
          const session: Session = {
            id: response.session_id,
            name: response.name,
            message_count: 0,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          };

          set((state) => ({
            sessions: [...state.sessions, session],
            currentSessionId: session.id,
            messages: {
              ...state.messages,
              [session.id]: [],
            },
          }));

          return session;
        } catch (error: unknown) {
          set({ error: (error as Error).message || '创建会话失败' });
          throw error;
        }
      },

      // 删除会话
      deleteSession: async (sessionId: string) => {
        try {
          await authService.deleteSession(sessionId);

          set((state) => {
            const newSessions = state.sessions.filter((s) => s.id !== sessionId);
            const newMessages = { ...state.messages };
            delete newMessages[sessionId];

            // 如果删除的是当前会话，切换到第一个会话
            let newCurrentSessionId = state.currentSessionId;
            if (state.currentSessionId === sessionId && newSessions.length > 0) {
              newCurrentSessionId = newSessions[0].id;
            } else if (newSessions.length === 0) {
              newCurrentSessionId = null;
            }

            return {
              sessions: newSessions,
              currentSessionId: newCurrentSessionId,
              messages: newMessages,
            };
          });
        } catch (error: unknown) {
          set({ error: (error as Error).message || '删除会话失败' });
          throw error;
        }
      },

      // 切换会话
      switchSession: (sessionId: string) => {
        set({ currentSessionId: sessionId });
      },

      // 加载会话列表
      loadSessions: async () => {
        try {
          const sessions = await authService.getSessions();

          // 如果没有会话，创建一个默认会话
          if (sessions.length === 0) {
            const newSession = await get().createSession();
            sessions.push(newSession);
          }

          set({
            sessions,
            currentSessionId: get().currentSessionId || sessions[0]?.id || null,
          });
        } catch (error: unknown) {
          set({ error: (error as Error).message || '加载会话失败' });
          throw error;
        }
      },

      // 发送消息
      sendMessage: async (content: string) => {
        const { currentSessionId, messages } = get();

        if (!currentSessionId) {
          throw new Error('没有选中的会话');
        }

        // 添加用户消息
        const userMessage: Message = {
          id: `msg-${Date.now()}`,
          role: 'user',
          content,
          timestamp: new Date().toISOString(),
        };

        set((state) => ({
          messages: {
            ...state.messages,
            [currentSessionId]: [...(state.messages[currentSessionId] || []), userMessage],
          },
          isStreaming: true,
        }));

        try {
          // 调用流式聊天 API
          await chatService.streamChat(
            {
              message: content,
              session_id: currentSessionId,
            },
            {
              onMessage: (data) => {
                const event = JSON.parse(data);

                if (event.event === 'token' || event.event === 'update') {
                  const newContent = event.data.content || '';

                  set((state) => {
                    const currentMessages = state.messages[currentSessionId] || [];
                    const lastMessage = currentMessages[currentMessages.length - 1];

                    // 如果最后一条消息是助手消息，更新内容
                    if (lastMessage && lastMessage.role === 'assistant') {
                      return {
                        messages: {
                          ...state.messages,
                          [currentSessionId]: [
                            ...currentMessages.slice(0, -1),
                            {
                              ...lastMessage,
                              content: lastMessage.content + newContent,
                            },
                          ],
                        },
                      };
                    }

                    // 否则添加新的助手消息
                    return {
                      messages: {
                        ...state.messages,
                        [currentSessionId]: [
                          ...currentMessages,
                          {
                            id: `msg-${Date.now()}`,
                            role: 'assistant',
                            content: newContent,
                            timestamp: new Date().toISOString(),
                          },
                        ],
                      },
                    };
                  });
                } else if (event.event === 'done') {
                  set({ isStreaming: false });
                }
              },
              onError: (error) => {
                set({ error: '消息发送失败', isStreaming: false });
              },
            }
          );
        } catch (error: unknown) {
          set({
            error: (error as Error).message || '发送消息失败',
            isStreaming: false,
          });
          throw error;
        }
      },

      // 添加消息
      addMessage: (sessionId: string, message: Message) => {
        set((state) => ({
          messages: {
            ...state.messages,
            [sessionId]: [...(state.messages[sessionId] || []), message],
          },
        }));
      },

      // 更新最后一条消息
      updateLastMessage: (sessionId: string, content: string) => {
        set((state) => {
          const currentMessages = state.messages[sessionId] || [];
          if (currentMessages.length === 0) return state;

          const lastMessage = currentMessages[currentMessages.length - 1];
          if (lastMessage.role !== 'assistant') return state;

          return {
            messages: {
              ...state.messages,
              [sessionId]: [
                ...currentMessages.slice(0, -1),
                { ...lastMessage, content },
              ],
            },
          };
        });
      },

      // 加载消息历史
      loadMessages: async (sessionId: string) => {
        set({ isLoading: true });

        try {
          const response = await chatService.getChatHistory(sessionId);

          set({
            messages: {
              ...get().messages,
              [sessionId]: response.messages.map((msg) => ({
                id: `msg-${Date.now()}-${Math.random()}`,
                role: msg.role,
                content: msg.content,
                timestamp: new Date().toISOString(),
              })),
            },
            isLoading: false,
          });
        } catch (error: unknown) {
          set({
            error: (error as Error).message || '加载消息失败',
            isLoading: false,
          });
          throw error;
        }
      },

      // 清除消息
      clearMessages: async (sessionId: string) => {
        try {
          await chatService.clearChatHistory(sessionId);

          set((state) => ({
            messages: {
              ...state.messages,
              [sessionId]: [],
            },
          }));
        } catch (error: unknown) {
          set({ error: (error as Error).message || '清除消息失败' });
          throw error;
        }
      },

      // 开始流式传输
      startStreaming: () => {
        set({ isStreaming: true });
      },

      // 停止流式传输
      stopStreaming: () => {
        set({ isStreaming: false });
      },

      // 清除错误
      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'kiki-chat-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        sessions: state.sessions,
        currentSessionId: state.currentSessionId,
        messages: state.messages,
      }),
    }
  )
);

export default useChatStore;
