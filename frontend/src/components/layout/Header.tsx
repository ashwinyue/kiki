/**
 * Kiki Agent Framework - 头部组件
 */

import { useAuth } from '@/hooks/useAuth';
import { useUIStore } from '@/stores';
import { classNames } from '@/utils/classNames';

export function Header() {
  const { user, logout } = useAuth();
  const { theme, toggleTheme, sidebarState, setSidebarState } = useUIStore();

  return (
    <header className="header">
      <div className="header-left">
        <button
          className="header-menu-button"
          onClick={() => {
            const newState = sidebarState === 'expanded' ? 'collapsed' : 'expanded';
            setSidebarState(newState);
          }}
          aria-label="切换侧边栏"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path
              d="M4 6H20M4 12H20M4 18H20"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            />
          </svg>
        </button>
        <div className="header-logo">Kiki Agent</div>
      </div>

      <div className="header-right">
        <button
          className="header-icon-button"
          onClick={toggleTheme}
          aria-label="切换主题"
        >
          {theme === 'light' ? (
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path
                d="M10 2C5.58172 2 2 5.58172 2 10C2 14.4183 5.58172 18 10 18C14.4183 18 18 14.4183 18 10"
                stroke="currentColor"
                strokeWidth="2"
              />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <circle cx="10" cy="10" r="4" stroke="currentColor" strokeWidth="2" />
              <path
                d="M10 2V4M10 16V18M18 10H16M4 10H2M15.66 15.66L14.24 14.24M5.76 5.76L4.34 4.34M15.66 4.34L14.24 5.76M5.76 14.24L4.34 15.66"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          )}
        </button>

        <div className="header-user">
          <span className="header-user-name">{user?.full_name || user?.email}</span>
          <button
            className="header-logout-button"
            onClick={logout}
            aria-label="退出登录"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path
                d="M6 4H4C2.89543 4 2 4.89543 2 6V14C2 15.1046 2.89543 16 4 16H6M12 4H14C15.1046 4 16 4.89543 16 6V14C16 15.1046 15.1046 16 14 16H12M9 9L12 12M12 12L9 15M12 12H2"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
}
