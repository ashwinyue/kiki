/**
 * Kiki Agent Framework - 设置页面
 */

import { useAuth } from '@/hooks/useAuth';
import { useTheme } from '@/hooks/useTheme';
import { useUIStore } from '@/stores';
import { Button } from '@/components/Button';

export function SettingsPage() {
  const { user, logout } = useAuth();
  const { theme, setTheme } = useTheme();
  const { sidebarState, setSidebarState } = useUIStore();

  return (
    <div className="page settings-page">
      <div className="page-header">
        <h1 className="page-title">设置</h1>
      </div>

      <div className="settings-sections">
        {/* 用户信息 */}
        <section className="settings-section">
          <h2 className="settings-section-title">用户信息</h2>
          <div className="settings-field">
            <label className="settings-label">邮箱</label>
            <div className="settings-value">{user?.email}</div>
          </div>
          <div className="settings-field">
            <label className="settings-label">姓名</label>
            <div className="settings-value">{user?.full_name || '未设置'}</div>
          </div>
        </section>

        {/* 外观设置 */}
        <section className="settings-section">
          <h2 className="settings-section-title">外观</h2>
          <div className="settings-field">
            <label className="settings-label">主题</label>
            <div className="settings-theme-selector">
              <button
                className={`theme-option ${theme === 'light' ? 'active' : ''}`}
                onClick={() => setTheme('light')}
              >
                <span className="theme-icon">☀️</span>
                <span>浅色</span>
              </button>
              <button
                className={`theme-option ${theme === 'dark' ? 'active' : ''}`}
                onClick={() => setTheme('dark')}
              >
                <span className="theme-icon">🌙</span>
                <span>深色</span>
              </button>
            </div>
          </div>
        </section>

        {/* 布局设置 */}
        <section className="settings-section">
          <h2 className="settings-section-title">布局</h2>
          <div className="settings-field">
            <label className="settings-label">侧边栏</label>
            <div className="settings-selector">
              <select
                className="settings-select"
                value={sidebarState}
                onChange={(e) => setSidebarState(e.target.value as any)}
              >
                <option value="expanded">展开</option>
                <option value="collapsed">收起</option>
                <option value="hidden">隐藏</option>
              </select>
            </div>
          </div>
        </section>

        {/* 危险操作 */}
        <section className="settings-section settings-section-danger">
          <h2 className="settings-section-title">危险操作</h2>
          <Button variant="primary" danger onClick={logout}>
            退出登录
          </Button>
        </section>
      </div>
    </div>
  );
}
