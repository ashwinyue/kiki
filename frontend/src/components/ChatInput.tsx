/**
 * Kiki Agent Framework - 聊天输入框组件
 *
 * 参考 WeKnora 的聊天输入框设计
 */
import React, { useRef, useState, KeyboardEvent, ClipboardEvent } from 'react';
import { classNames } from '@/utils/classNames';

export interface ChatInputProps {
  /** 占位符文本 */
  placeholder?: string;
  /** 禁用状态 */
  disabled?: boolean;
  /** 最大输入长度 */
  maxLength?: number;
  /** 发送回调 */
  onSend: (content: string) => void;
  /** 输入变化回调 */
  onChange?: (content: string) => void;
  /** 粘贴回调（支持文件） */
  onPaste?: (event: ClipboardEvent<HTMLTextAreaElement>) => void;
  /** 是否正在发送 */
  loading?: boolean;
  /** 自定义类名 */
  className?: string;
  /** 支持的文件类型 */
  acceptFiles?: string;
}

/**
 * 聊天输入框组件
 */
export const ChatInput: React.FC<ChatInputProps> = ({
  placeholder = '输入消息...',
  disabled = false,
  maxLength = 5000,
  onSend,
  onChange,
  onPaste,
  loading = false,
  className,
  acceptFiles = 'image/*,.pdf,.doc,.docx,.txt',
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [content, setContent] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [isComposing, setIsComposing] = useState(false);

  // 自动调整高度
  const adjustHeight = React.useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = 'auto';
    const scrollHeight = textarea.scrollHeight;
    const maxHeight = 200; // 最大高度
    textarea.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
  }, []);

  // 处理输入变化
  const handleChange = (value: string) => {
    if (value.length > maxLength) {
      value = value.slice(0, maxLength);
    }
    setContent(value);
    onChange?.(value);
    adjustHeight();
  };

  // 处理键盘事件
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter 发送，Shift+Enter 换行
    if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
      e.preventDefault();
      handleSend();
    }
  };

  // 发送消息
  const handleSend = () => {
    const trimmed = content.trim();
    if (!trimmed || disabled || loading) return;

    onSend(trimmed);
    setContent('');
    adjustHeight();

    // 重新聚焦
    setTimeout(() => {
      textareaRef.current?.focus();
    }, 100);
  };

  // 处理粘贴
  const handlePaste = (e: ClipboardEvent<HTMLTextAreaElement>) => {
    onPaste?.(e);
  };

  // 字符计数
  const charCount = content.length;
  const isNearLimit = charCount > maxLength * 0.9;
  const canSend = content.trim().length > 0 && !loading && !disabled;

  return (
    <div
      className={classNames(
        'chat-input-container',
        isFocused && 'chat-input-focused',
        disabled && 'chat-input-disabled',
        className
      )}
    >
      {/* 输入区域 */}
      <div className="chat-input-wrapper">
        <textarea
          ref={textareaRef}
          className="chat-input-textarea"
          placeholder={placeholder}
          value={content}
          onChange={(e) => handleChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          disabled={disabled}
          rows={1}
        />

        {/* 底部工具栏 */}
        <div className="chat-input-toolbar">
          {/* 左侧：字符计数 */}
          <div
            className={classNames(
              'char-count',
              isNearLimit && 'char-count-warning'
            )}
          >
            {charCount} / {maxLength}
          </div>

          {/* 右侧：发送按钮 */}
          <button
            type="button"
            className={classNames(
              'send-button',
              canSend && 'send-button-active'
            )}
            onClick={handleSend}
            disabled={!canSend}
            aria-label="发送消息"
          >
            {loading ? (
              <span className="loading-spinner" />
            ) : (
              <svg
                width="20"
                height="20"
                viewBox="0 0 20 20"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M2.5 10L17.5 10L10 3.5M17.5 10L10 16.5"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* 提示信息 */}
      <div className="chat-input-hint">
        按 <kbd>Enter</kbd> 发送，<kbd>Shift + Enter</kbd> 换行
      </div>
    </div>
  );
};

export default ChatInput;
