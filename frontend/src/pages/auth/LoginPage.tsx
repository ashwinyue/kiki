/**
 * Kiki Agent Framework - 登录页面
 */

import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { Button } from '@/components/Button';

export function LoginPage() {
  const navigate = useNavigate();
  const { login, isLoading, error, clearError } = useAuth();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      await login(username, password);
      navigate('/chat');
    } catch (err) {
      // 错误已在 store 中处理
    }
  };

  return (
    <div className="auth-page">
      <form className="auth-form" onSubmit={handleSubmit}>
        <h2 className="auth-title">登录</h2>

        {error && (
          <div className="auth-error">
            {error}
            <button
              type="button"
              className="auth-error-close"
              onClick={clearError}
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

        <div className="auth-field">
          <label htmlFor="username" className="auth-label">
            邮箱
          </label>
          <input
            id="username"
            type="email"
            className="auth-input"
            placeholder="请输入邮箱"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            disabled={isLoading}
          />
        </div>

        <div className="auth-field">
          <label htmlFor="password" className="auth-label">
            密码
          </label>
          <input
            id="password"
            type="password"
            className="auth-input"
            placeholder="请输入密码"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            disabled={isLoading}
            minLength={8}
          />
        </div>

        <Button
          type="submit"
          variant="primary"
          size="large"
          block
          loading={isLoading}
        >
          登录
        </Button>

        <div className="auth-footer">
          还没有账号？{' '}
          <Link to="/auth/register" className="auth-link">
            立即注册
          </Link>
        </div>
      </form>
    </div>
  );
}
