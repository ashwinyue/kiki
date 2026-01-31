/**
 * Kiki Agent Framework - 应用根组件
 */

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

// 布局组件
import { MainLayout } from './components/layout/MainLayout';
import { AuthLayout } from './components/layout/AuthLayout';

// 页面组件
import { ChatPage } from './pages/chat/ChatPage';
import { LoginPage } from './pages/auth/LoginPage';
import { RegisterPage } from './pages/auth/RegisterPage';
import { AgentsPage } from './pages/agents/AgentsPage';
import { ToolsPage } from './pages/tools/ToolsPage';
import { SettingsPage } from './pages/settings/SettingsPage';

/**
 * 应用根组件
 * 注意：当前版本跳过登录验证，直接访问所有页面
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* 认证相关路由 */}
        <Route path="/auth" element={<AuthLayout />}>
          <Route path="login" element={<LoginPage />} />
          <Route path="register" element={<RegisterPage />} />
          <Route index element={<Navigate to="/auth/login" replace />} />
        </Route>

        {/* 主应用路由 - 无需认证 */}
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Navigate to="/chat" replace />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="agents" element={<AgentsPage />} />
          <Route path="tools" element={<ToolsPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
