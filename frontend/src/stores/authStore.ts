/**
 * Kiki Agent Framework - 认证状态管理
 *
 * 使用 Zustand 进行轻量级状态管理
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { User } from '@/types/auth';
import { auth as authService } from '@/services';

/**
 * 认证状态接口
 */
interface AuthState {
  // 状态
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  login: (username: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => void;
  fetchUser: () => Promise<void>;
  clearError: () => void;
}

/**
 * 认证 Store
 */
export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      // 初始状态
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      // 登录
      login: async (username: string, password: string) => {
        set({ isLoading: true, error: null });

        try {
          const response = await authService.login({ username, password });

          // 存储 Token 到 localStorage
          if (typeof window !== 'undefined') {
            localStorage.setItem('access_token', response.access_token);
          }

          // 存储用户信息和 Token
          set({
            user: {
              id: 0, // 会在 fetchUser 中更新
              email: username,
              full_name: null,
              is_active: true,
              is_superuser: false,
            },
            isAuthenticated: true,
            isLoading: false,
          });

          // 获取完整用户信息
          await get().fetchUser();
        } catch (error: unknown) {
          set({
            error: (error as Error).message || '登录失败',
            isLoading: false,
          });
          throw error;
        }
      },

      // 注册
      register: async (email: string, password: string, fullName?: string) => {
        set({ isLoading: true, error: null });

        try {
          const response = await authService.register({
            email,
            password,
            full_name: fullName,
          });

          // 存储 Token 到 localStorage
          if (typeof window !== 'undefined') {
            localStorage.setItem('access_token', response.access_token);
          }

          set({
            user: response,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: unknown) {
          set({
            error: (error as Error).message || '注册失败',
            isLoading: false,
          });
          throw error;
        }
      },

      // 登出
      logout: () => {
        if (typeof window !== 'undefined') {
          localStorage.removeItem('access_token');
        }
        set({
          user: null,
          isAuthenticated: false,
          error: null,
        });
      },

      // 获取用户信息
      fetchUser: async () => {
        try {
          const user = await authService.getCurrentUser();
          set({ user, isAuthenticated: user.is_active });
        } catch (error) {
          // Token 可能过期，清除认证状态
          set({
            user: null,
            isAuthenticated: false,
          });
        }
      },

      // 清除错误
      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'kiki-auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

export default useAuthStore;
