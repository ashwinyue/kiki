/**
 * Kiki Agent Framework - 认证布局组件
 */

import { Outlet } from 'react-router-dom';

export function AuthLayout() {
  return (
    <div className="auth-layout">
      <div className="auth-container">
        <div className="auth-logo">
          <h1>Kiki Agent</h1>
          <p>企业级 AI Agent 开发平台</p>
        </div>
        <Outlet />
      </div>
    </div>
  );
}
