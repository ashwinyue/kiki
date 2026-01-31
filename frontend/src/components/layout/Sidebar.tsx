/**
 * Kiki Agent Framework - 侧边栏组件
 */

import { NavLink } from 'react-router-dom';
import { useChat } from '@/hooks/useChat';
import { classNames } from '@/utils/classNames';
import { useUIStore } from '@/stores';

interface SidebarLinkProps {
  to: string;
  icon: React.ReactNode;
  label: string;
}

function SidebarLink({ to, icon, label }: SidebarLinkProps) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        classNames('sidebar-link', isActive && 'sidebar-link-active')
      }
    >
      <span className="sidebar-link-icon">{icon}</span>
      <span className="sidebar-link-label">{label}</span>
    </NavLink>
  );
}

export function Sidebar() {
  const { sessions, currentSessionId, switchSession, createSession } = useChat();
  const { sidebarState } = useUIStore();
  const isCollapsed = sidebarState === 'collapsed';

  return (
    <aside className={classNames('sidebar', isCollapsed && 'sidebar-collapsed')}>
      {/* 导航菜单 */}
      <nav className="sidebar-nav">
        <SidebarLink
          to="/chat"
          icon={
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path
                d="M10 2C5.58172 2 2 5.58172 2 10C2 14.4183 5.58172 18 10 18C14.4183 18 18 14.4183 18 10"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          }
          label="对话"
        />
        <SidebarLink
          to="/agents"
          icon={
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path
                d="M10 2C8.89543 2 8 2.89543 8 4V6H12V4C12 2.89543 11.1046 2 10 2Z"
                fill="currentColor"
              />
              <path
                d="M10 8V12M10 12L8 10M10 12L12 10"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="2" />
            </svg>
          }
          label="Agents"
        />
        <SidebarLink
          to="/tools"
          icon={
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path
                d="M2 10H18M10 2V18"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
              <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="2" />
            </svg>
          }
          label="工具"
        />
        <SidebarLink
          to="/settings"
          icon={
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="2" />
              <path
                d="M10 5V10L13 13"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          }
          label="设置"
        />
      </nav>

      {/* 会话列表 */}
      {!isCollapsed && (
        <div className="sidebar-sessions">
          <div className="sidebar-sessions-header">
            <span className="sidebar-sessions-title">会话</span>
            <button
              className="sidebar-new-chat-button"
              onClick={() => createSession('新对话')}
              aria-label="新建对话"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 3V13M3 8H13" stroke="currentColor" strokeWidth="2" />
              </svg>
            </button>
          </div>
          <div className="sidebar-sessions-list">
            {sessions.map((session) => (
              <button
                key={session.id}
                className={classNames(
                  'sidebar-session-item',
                  session.id === currentSessionId && 'sidebar-session-item-active'
                )}
                onClick={() => switchSession(session.id)}
              >
                <span className="sidebar-session-name">{session.name}</span>
                <span className="sidebar-session-count">{session.message_count}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}
