/**
 * Kiki Agent Framework - 认证 Hook
 */

import { useCallback } from 'react';
import { useAuthStore } from '@/stores';

export interface UseAuthReturn {
  // 状态
  user: {
    id: number;
    email: string;
    full_name: string | null;
    is_active: boolean;
    is_superuser: boolean;
  } | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // 操作
  login: (username: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
  clearError: () => void;
}

/**
 * 认证 Hook
 */
export function useAuth(): UseAuthReturn {
  const {
    user,
    isAuthenticated,
    isLoading,
    error,
    login: storeLogin,
    register: storeRegister,
    logout: storeLogout,
    fetchUser,
    clearError: storeClearError,
  } = useAuthStore();

  // 登录
  const login = useCallback(
    async (username: string, password: string) => {
      await storeLogin(username, password);
    },
    [storeLogin]
  );

  // 注册
  const register = useCallback(
    async (email: string, password: string, fullName?: string) => {
      await storeRegister(email, password, fullName);
    },
    [storeRegister]
  );

  // 登出
  const logout = useCallback(() => {
    storeLogout();
    localStorage.removeItem('access_token');
  }, [storeLogout]);

  // 刷新用户信息
  const refreshUser = useCallback(async () => {
    await fetchUser();
  }, [fetchUser]);

  // 清除错误
  const clearError = useCallback(() => {
    storeClearError();
  }, [storeClearError]);

  return {
    user,
    isAuthenticated,
    isLoading,
    error,
    login,
    register,
    logout,
    refreshUser,
    clearError,
  };
}

export default useAuth;
