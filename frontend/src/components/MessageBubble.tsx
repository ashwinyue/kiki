/**
 * Kiki Agent Framework - 消息气泡组件
 *
 * 参考 WeKnora 的消息气泡设计
 */
import React from 'react';
import { classNames } from '@/utils/classNames';

export interface MessageBubbleProps {
  /** 消息内容 */
  content: string;
  /** 消息类型 */
  type: 'user' | 'assistant' | 'system';
  /** 是否正在输入（显示动画） */
  isTyping?: boolean;
  /** 提及的知识库和文件 */
  mentionedItems?: MentionedItem[];
  /** 自定义类名 */
  className?: string;
}

export interface MentionedItem {
  id: string;
  name: string;
  type: 'kb' | 'faq' | 'file' | 'agent';
  kb_type?: 'doc' | 'faq';
}

/**
 * 提及标签组件
 */
const MentionTag: React.FC<{ item: MentionedItem }> = ({ item }) => {
  const tagClass = React.useMemo(() => {
    switch (item.type) {
      case 'kb':
        return item.kb_type === 'faq' ? 'tag-faq' : 'tag-kb';
      case 'faq':
        return 'tag-faq';
      case 'file':
        return 'tag-file';
      case 'agent':
        return 'tag-agent';
      default:
        return 'tag-kb';
    }
  }, [item.type, item.kb_type]);

  const icon = React.useMemo(() => {
    switch (item.type) {
      case 'kb':
        return item.kb_type === 'faq' ? '❓' : '📁';
      case 'faq':
        return '❓';
      case 'file':
        return '📄';
      case 'agent':
        return '🤖';
      default:
        return '📎';
    }
  }, [item.type, item.kb_type]);

  return (
    <span className={classNames('mention-tag', tagClass)}>
      <span className="tag-icon">{icon}</span>
      <span className="tag-name">{item.name}</span>
    </span>
  );
};

/**
 * 正在输入动画组件
 */
const TypingIndicator: React.FC = () => {
  return (
    <div className="typing-indicator">
      <span className="typing-dot" />
      <span className="typing-dot" style={{ animationDelay: '0.2s' }} />
      <span className="typing-dot" style={{ animationDelay: '0.4s' }} />
    </div>
  );
};

/**
 * 消息气泡组件
 */
export const MessageBubble: React.FC<MessageBubbleProps> = ({
  content,
  type,
  isTyping = false,
  mentionedItems = [],
  className,
}) => {
  const isUser = type === 'user';
  const isSystem = type === 'system';

  if (isSystem) {
    return (
      <div className={classNames('message-bubble', 'message-system', className)}>
        <span className="system-message">{content}</span>
      </div>
    );
  }

  return (
    <div
      className={classNames(
        'message-bubble',
        isUser ? 'message-user' : 'message-assistant',
        className
      )}
    >
      {/* 提及标签 */}
      {!isUser && mentionedItems.length > 0 && (
        <div className="mentioned-items">
          {mentionedItems.map((item) => (
            <MentionTag key={item.id} item={item} />
          ))}
        </div>
      )}

      {/* 消息内容 */}
      {isTyping ? (
        <TypingIndicator />
      ) : (
        <div
          className={classNames(
            'message-content',
            isUser && 'user-message-content'
          )}
        >
          {content}
        </div>
      )}
    </div>
  );
};

export default MessageBubble;
