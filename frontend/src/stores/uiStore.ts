/**
 * Kiki Agent Framework - UI 状态管理
 *
 * 使用 Zustand 进行轻量级状态管理
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

/**
 * 主题类型
 */
export type Theme = 'light' | 'dark';

/**
 * 侧边栏状态
 */
export type SidebarState = 'expanded' | 'collapsed' | 'hidden';

/**
 * UI 状态接口
 */
interface UIState {
  // 状态
  theme: Theme;
  sidebarState: SidebarState;
  isMobile: boolean;
  notification: {
    show: boolean;
    message: string;
    type: 'success' | 'error' | 'warning' | 'info';
  } | null;

  // Actions - 主题
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;

  // Actions - 侧边栏
  setSidebarState: (state: SidebarState) => void;
  toggleSidebar: () => void;

  // Actions - 响应式
  setIsMobile: (isMobile: boolean) => void;

  // Actions - 通知
  showNotification: (
    message: string,
    type?: 'success' | 'error' | 'warning' | 'info'
  ) => void;
  hideNotification: () => void;
}

/**
 * UI Store
 */
export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      // 初始状态
      theme: (localStorage.getItem('theme') as Theme) || 'light',
      sidebarState: 'expanded',
      isMobile: false,
      notification: null,

      // 设置主题
      setTheme: (theme: Theme) => {
        set({ theme });
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
      },

      // 切换主题
      toggleTheme: () => {
        const currentTheme = get().theme;
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        get().setTheme(newTheme);
      },

      // 设置侧边栏状态
      setSidebarState: (state: SidebarState) => {
        set({ sidebarState: state });
      },

      // 切换侧边栏
      toggleSidebar: () => {
        const currentState = get().sidebarState;
        let newState: SidebarState;

        if (currentState === 'expanded') {
          newState = 'collapsed';
        } else if (currentState === 'collapsed') {
          newState = 'expanded';
        } else {
          newState = 'expanded';
        }

        set({ sidebarState: newState });
      },

      // 设置移动端状态
      setIsMobile: (isMobile: boolean) => {
        set({ isMobile });
        // 移动端默认隐藏侧边栏
        if (isMobile && get().sidebarState !== 'hidden') {
          set({ sidebarState: 'hidden' });
        }
      },

      // 显示通知
      showNotification: (
        message: string,
        type: 'success' | 'error' | 'warning' | 'info' = 'info'
      ) => {
        set({ notification: { show: true, message, type } });

        // 3 秒后自动隐藏
        setTimeout(() => {
          get().hideNotification();
        }, 3000);
      },

      // 隐藏通知
      hideNotification: () => {
        set({ notification: null });
      },
    }),
    {
      name: 'kiki-ui-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        theme: state.theme,
        sidebarState: state.sidebarState,
      }),
    }
  )
);

// 初始化主题
if (typeof window !== 'undefined') {
  const theme = localStorage.getItem('theme') as Theme;
  if (theme) {
    document.documentElement.setAttribute('data-theme', theme);
  }
}

export default useUIStore;
