/**
 * Kiki Agent Framework - 聊天主区域
 */

import { useChat } from '@/hooks/useChat';
import { MessageBubble } from '@/components/MessageBubble';
import { ChatInput } from '@/components/ChatInput';
import { classNames } from '@/utils/classNames';
import { useEffect, useRef } from 'react';

export function ChatMain() {
  const {
    currentMessages,
    isStreaming,
    sendMessage,
    clearError,
    error,
  } = useChat();

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentMessages]);

  const handleSend = async (content: string) => {
    try {
      await sendMessage(content);
    } catch (err) {
      // 错误已在 store 中处理
    }
  };

  return (
    <main className="chat-main">
      {/* 消息列表 */}
      <div className="chat-messages">
        {currentMessages.length === 0 ? (
          <div className="chat-empty">
            <div className="chat-empty-icon">💬</div>
            <h3 className="chat-empty-title">开始对话</h3>
            <p className="chat-empty-description">
              输入消息开始与 Kiki Agent 交流
            </p>
          </div>
        ) : (
          currentMessages.map((message, index) => (
            <MessageBubble
              key={message.id || index}
              content={message.content}
              type={message.role}
              isTyping={false}
            />
          ))
        )}
        {isStreaming && (
          <MessageBubble
            content=""
            type="assistant"
            isTyping={true}
          />
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="chat-error">
          <span className="chat-error-message">{error}</span>
          <button
            className="chat-error-close"
            onClick={clearError}
            aria-label="关闭"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path
                d="M10.5 3.5L3.5 10.5M3.5 3.5L10.5 10.5"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>
      )}

      {/* 输入框 */}
      <div className="chat-input-wrapper">
        <ChatInput
          placeholder="输入消息..."
          onSend={handleSend}
          loading={isStreaming}
        />
      </div>
    </main>
  );
}
