/**
 * Kiki Agent Framework - 主题 Hook
 */

import { useCallback, useEffect } from 'react';
import { useUIStore } from '@/stores';
import type { Theme } from '@/stores/uiStore';

export interface UseThemeReturn {
  theme: Theme;
  isDark: boolean;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

/**
 * 主题 Hook
 */
export function useTheme(): UseThemeReturn {
  const { theme, setTheme: storeSetTheme, toggleTheme: storeToggleTheme } = useUIStore();

  // 设置主题
  const setTheme = useCallback(
    (newTheme: Theme) => {
      storeSetTheme(newTheme);
    },
    [storeSetTheme]
  );

  // 切换主题
  const toggleTheme = useCallback(() => {
    storeToggleTheme();
  }, [storeToggleTheme]);

  // 初始化主题
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as Theme;
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = savedTheme || (prefersDark ? 'dark' : 'light');

    if (initialTheme !== theme) {
      setTheme(initialTheme);
    }
  }, [theme, setTheme]);

  return {
    theme,
    isDark: theme === 'dark',
    setTheme,
    toggleTheme,
  };
}

export default useTheme;
