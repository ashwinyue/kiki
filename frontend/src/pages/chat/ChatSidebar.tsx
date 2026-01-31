/**
 * Kiki Agent Framework - 聊天侧边栏
 */

import { useChat } from '@/hooks/useChat';
import { classNames } from '@/utils/classNames';

export function ChatSidebar() {
  const {
    sessions,
    currentSessionId,
    switchSession,
    createSession,
    deleteSession,
  } = useChat();

  return (
    <aside className="chat-sidebar">
      <div className="chat-sidebar-header">
        <h2 className="chat-sidebar-title">对话</h2>
        <button
          className="chat-new-button"
          onClick={() => createSession('新对话')}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M8 3V13M3 8H13" stroke="currentColor" strokeWidth="2" />
          </svg>
          新对话
        </button>
      </div>

      <div className="chat-sidebar-list">
        {sessions.map((session) => (
          <div
            key={session.id}
            className={classNames(
              'chat-session-item',
              session.id === currentSessionId && 'chat-session-item-active'
            )}
          >
            <button
              className="chat-session-button"
              onClick={() => switchSession(session.id)}
            >
              <span className="chat-session-name">{session.name}</span>
              <span className="chat-session-count">{session.message_count}</span>
            </button>
            <button
              className="chat-session-delete"
              onClick={() => deleteSession(session.id)}
              aria-label="删除会话"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path
                  d="M11 3.5H3M3 3.5L3.5 12C3.5 12.2761 3.72386 12.5 4 12.5H10C10.2761 12.5 10.5 12.2761 10.5 12L11 3.5Z"
                  stroke="currentColor"
                  strokeWidth="1"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
