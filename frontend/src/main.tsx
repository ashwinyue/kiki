/**
 * Kiki Agent Framework - 应用入口
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './theme/index.css';

// 初始化主题
const initTheme = () => {
  const savedTheme = localStorage.getItem('theme') as 'light' | 'dark';
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = savedTheme || (prefersDark ? 'dark' : 'light');

  document.documentElement.setAttribute('data-theme', theme);
};

initTheme();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
