# Kiki Agent Framework - 前端架构设计

> 版本: v0.1.0
> 设计理念: "流动的智能" (Fluid Intelligence)
> 更新日期: 2025-01-31

---

## 目录

- [设计概览](#设计概览)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [设计系统](#设计系统)
- [核心组件](#核心组件)
- [状态管理](#状态管理)
- [路由设计](#路由设计)
- [API 客户端](#api-客户端)
- [样式方案](#样式方案)
- [构建配置](#构建配置)

---

## 设计概览

### 设计理念：流动的智能

Kiki 前端采用独特的设计语言，区别于 WeKnora 的绿色主题，打造温暖而专业的视觉体验：

| 设计维度 | Kiki 独特方案 |
|---------|-------------|
| **主色调** | **琥珀橙** `#f59e0b` → 唤醒、温暖、创造力 |
| **辅助色** | **靛蓝** `#6366f1` → 深度思考、逻辑 |
| **字体** | `Geist` (显示) + `IBM Plex Sans` (正文) |
| **风格** | 玻璃态 + 流动渐变 + 微妙噪点 |
| **布局** | 非对称卡片 + 悬浮层次 |

### 视觉特点

```
┌─────────────────────────────────────────────────────────────┐
│                    设计语言核心要素                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🎨 琥珀橙渐变        → 暖色调，激发创造力                   │
│  🔷 靛蓝辅助        → 理性思考，深度逻辑                   │
│  🪟 玻璃态卡片        → 通透感，现代科技                     │
│  ✨ 流动渐变背景      → 模拟思维流动                        │
│  🌫️ 微妙噪点纹理      → 增加质感，避免单调                   │
│                                                             │
│  圆角: 12px (大圆角)                                        │
│  阴影: 多层深度 + 有色光晕                                   │
│  动画: 弹性缓动 + 微交互反馈                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 与 WeKnora 的设计差异

| 方面 | WeKnora | Kiki |
|-----|---------|------|
| **主色调** | 绿色 `#07c05f` | 琥珀橙 `#f59e0b` |
| **辅助色** | 无 | 靛蓝 `#6366f1` |
| **用户消息** | 浅绿纯色 `#8CE97F` | 琥珀橙渐变 |
| **字体** | PingFang SC | Geist + IBM Plex Sans |
| **背景** | 纯色 | 流动渐变网格 + 噪点 |
| **卡片** | 纯色 + 边框 | 玻璃态 + 模糊 |
| **动画** | 基础缓动 | 弹性缓动 + 微交互 |

---

## 技术栈

### 核心框架

| 技术 | 版本 | 用途 |
|-----|------|------|
| **React** | 18.x | UI 框架 |
| **TypeScript** | 5.x | 类型安全 |
| **Vite** | 5.x | 构建工具 |

### 路由和状态

| 技术 | 版本 | 用途 |
|-----|------|------|
| **React Router** | 6.x | 单页面路由 |
| **Zustand** | 4.x | 轻量状态管理 |
| **React Query** | 5.x | 服务器状态管理 |

### UI 效果

| 技术 | 版本 | 用途 |
|-----|------|------|
| **Framer Motion** | 11.x | 高级动画 |
| **React Marked** | 12.x | Markdown 渲染 |
| **Highlight.js** | 11.x | 代码高亮 |
| **DOMPurify** | 3.x | XSS 防护 |

### 工具库

| 技术 | 版本 | 用途 |
|-----|------|------|
| **Axios** | 1.x | HTTP 客户端 |
| **dayjs** | 1.x | 日期处理 |
| **nanoid** | 5.x | ID 生成 |

---

## 项目结构

```
frontend/
├── src/
│   ├── App.tsx                 # 应用入口
│   ├── main.tsx                # React 挂载点
│   │
│   ├── pages/                  # 页面组件
│   │   ├── chat/               # 聊天页面
│   │   │   ├── ChatPage.tsx
│   │   │   ├── ChatSidebar.tsx
│   │   │   └── ChatMain.tsx
│   │   ├── agents/             # Agent 管理页
│   │   ├── tools/              # 工具管理页
│   │   ├── settings/           # 设置页
│   │   └── auth/               # 认证页
│   │
│   ├── components/             # 通用组件
│   │   ├── layout/             # 布局组件
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── MainLayout.tsx
│   │   ├── chat/               # 聊天组件
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── ChatInput.tsx
│   │   │   ├── MessageList.tsx
│   │   │   └── StreamingText.tsx
│   │   ├── ui/                 # UI 组件
│   │   │   ├── Button.tsx
│   │   │   ├── Input.tsx
│   │   │   ├── Modal.tsx
│   │   │   ├── Dropdown.tsx
│   │   │   └── Tag.tsx
│   │   └── index.ts
│   │
│   ├── hooks/                  # 自定义 Hooks
│   │   ├── useChat.ts          # 聊天 Hook
│   │   ├── useStream.ts        # SSE 流式 Hook
│   │   ├── useTheme.ts         # 主题 Hook
│   │   ├── useAuth.ts          # 认证 Hook
│   │   └── useLocalStorage.ts
│   │
│   ├── stores/                 # Zustand 状态
│   │   ├── chatStore.ts
│   │   ├── authStore.ts
│   │   ├── uiStore.ts
│   │   └── agentStore.ts
│   │
│   ├── services/               # API 服务
│   │   ├── api.ts
│   │   ├── chat.ts
│   │   ├── auth.ts
│   │   ├── agent.ts
│   │   └── sse.ts
│   │
│   ├── types/                  # TypeScript 类型
│   │   ├── chat.ts
│   │   ├── agent.ts
│   │   └── auth.ts
│   │
│   ├── utils/                  # 工具函数
│   │   ├── classNames.ts
│   │   ├── format.ts
│   │   └── validation.ts
│   │
│   ├── theme/                  # 主题系统
│   │   ├── index.css
│   │   ├── variables.css       # CSS 变量
│   │   ├── global.css
│   │   └── markdown.css
│   │
│   └── assets/                 # 静态资源
│       ├── icons/
│       ├── images/
│       └── fonts/
│
├── index.html
├── vite.config.ts
├── tsconfig.json
└── package.json
```

---

## 设计系统

### 颜色系统

#### 品牌色 - 琥珀橙系

```css
--kiki-amber-50:  #fff8eb;   /* 极浅 */
--kiki-amber-400: #fbbf24;   /* hover */
--kiki-amber-500: #f59e0b;   /* 主色 */
--kiki-amber-600: #d97706;   /* active */
```

#### 辅助色 - 靛蓝系

```css
--kiki-indigo-400: #818cf8;
--kiki-indigo-500: #6366f1;   /* 辅助主色 */
--kiki-indigo-600: #4f46e5;
```

#### 渐变定义

```css
/* 琥珀橙渐变 */
--kiki-gradient-amber: linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%);

/* 靛蓝渐变 */
--kiki-gradient-indigo: linear-gradient(135deg, #818cf8 0%, #6366f1 50%, #4f46e5 100%);

/* 流动网格背景 */
--kiki-gradient-mesh: radial-gradient(at 40% 20%, rgba(251, 191, 36, 0.15) 0px, transparent 50%),
                      radial-gradient(at 80% 0%, rgba(99, 102, 241, 0.1) 0px, transparent 50%),
                      radial-gradient(at 0% 50%, rgba(245, 158, 11, 0.08) 0px, transparent 50%);
```

#### 消息气泡颜色

```css
/* 用户消息 - 琥珀橙渐变 */
--kiki-user-msg-bg: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
--kiki-user-msg-text: #1c1917;

/* 助手消息 - 玻璃态卡片 */
--kiki-bot-msg-bg: #ffffff;
--kiki-bot-msg-text: #1c1917;
```

### 字体系统

```css
/* 显示字体 - 用于标题 */
--kiki-font-display: 'Geist', 'SF Pro Display', -apple-system, sans-serif;

/* 正文字体 - 用于正文 */
--kiki-font-body: 'IBM Plex Sans', 'SF Pro Text', -apple-system, sans-serif;

/* 等宽字体 - 用于代码 */
--kiki-font-mono: 'IBM Plex Mono', 'SF Mono', 'Consolas', monospace;
```

### 圆角系统

```css
--kiki-radius-sm:   4px;    /* 小元素 */
--kiki-radius-md:   8px;    /* 按钮 */
--kiki-radius-lg:   12px;   /* 卡片、输入框 */
--kiki-radius-xl:   16px;   /* 大卡片 */
--kiki-radius-2xl:  24px;   /* 容器 */
--kiki-radius-full: 9999px; /* 圆形 */
```

### 阴影系统

```css
/* 常规阴影 */
--kiki-shadow-sm:  0 1px 3px rgba(0, 0, 0, 0.08);
--kiki-shadow-md:  0 4px 6px rgba(0, 0, 0, 0.08);
--kiki-shadow-lg:  0 10px 15px rgba(0, 0, 0, 0.08);

/* 有色光晕 */
--kiki-shadow-amber:  0 8px 30px rgba(245, 158, 11, 0.2);
--kiki-shadow-indigo: 0 8px 30px rgba(99, 102, 241, 0.2);
```

### 玻璃态效果

```css
--kiki-glass-bg: rgba(255, 255, 255, 0.7);
--kiki-glass-border: rgba(255, 255, 255, 0.18);
--kiki-glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);

/* 应用到元素 */
.glass {
  background: var(--kiki-glass-bg);
  backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid var(--kiki-glass-border);
  box-shadow: var(--kiki-glass-shadow);
}
```

---

## 核心组件

### 消息气泡 (MessageBubble)

**设计特点**：
- 用户消息：琥珀橙渐变背景 + 右侧尾巴装饰
- 助手消息：玻璃态卡片 + 左侧机器人头像装饰
- 弹性入场动画 (scale + translateY)

```typescript
interface MessageBubbleProps {
  content: string;
  type: 'user' | 'assistant' | 'system';
  isTyping?: boolean;
  mentionedItems?: MentionedItem[];
}
```

### 聊天输入框 (ChatInput)

**设计特点**：
- 玻璃态容器
- 聚焦时琥珀橙光晕效果
- 琥珀橙渐变发送按钮

```typescript
interface ChatInputProps {
  placeholder?: string;
  disabled?: boolean;
  maxLength?: number;
  onSend: (content: string) => void;
  loading?: boolean;
}
```

### 按钮 (Button)

**设计特点**：
- 主要按钮：琥珀橙渐变 + 有色阴影
- 次要按钮：靛蓝渐变
- 流光 hover 效果

```typescript
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'text';
  size?: 'small' | 'medium' | 'large';
  shape?: 'square' | 'round' | 'circle';
  loading?: boolean;
  icon?: React.ReactNode;
}
```

---

## 状态管理

### Zustand Store 设计

```typescript
// stores/chatStore.ts
interface ChatStore {
  sessions: Session[];
  currentSessionId: string | null;
  messages: Record<string, Message[]>;
  isStreaming: boolean;

  createSession: (name: string) => string;
  sendMessage: (sessionId: string, content: string) => Promise<void>;
  clearMessages: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
}

export const useChatStore = create<ChatStore>((set, get) => ({
  // ... 实现
}));
```

---

## 路由设计

### 路由结构

```typescript
const routes = [
  {
    path: '/',
    element: <MainLayout />,
    children: [
      { path: '', element: <Navigate to="/chat" /> },
      {
        path: 'chat',
        element: <ChatPage />,
        children: [
          { path: ':sessionId', element: <ChatMain /> }
        ]
      },
      { path: 'agents', element: <AgentsPage /> },
      { path: 'tools', element: <ToolsPage /> },
      { path: 'settings', element: <SettingsPage /> },
    ]
  },
  {
    path: '/auth',
    element: <AuthLayout />,
    children: [
      { path: 'login', element: <LoginPage /> },
      { path: 'register', element: <RegisterPage /> },
    ]
  },
];
```

---

## API 客户端

### Axios 配置

```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
  timeout: 30000,
});

// 请求拦截器
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  async (error) => {
    if (error.response?.status === 401) {
      // Token 刷新逻辑
    }
    return Promise.reject(error);
  }
);
```

### SSE 流式客户端

```typescript
export async function streamChat(
  message: string,
  sessionId: string,
  onChunk: (chunk: string) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch('/api/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    // 解析 SSE 格式
  }
}
```

---

## 样式方案

### CSS Modules + CSS Variables

```tsx
import styles from './MessageBubble.module.css';

export const MessageBubble: React.FC<MessageBubbleProps> = ({ content, type }) => {
  return (
    <div className={styles[`message-${type}`]}>
      <div className={styles.content}>{content}</div>
    </div>
  );
};
```

### 主题切换

```typescript
export function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    return localStorage.getItem('theme') as 'light' | 'dark' || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  }, []);

  return { theme, toggleTheme, setTheme };
}
```

---

## 构建配置

### Vite 配置

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'utils': ['axios', 'dayjs', 'nanoid'],
        },
      },
    },
  },
});
```

---

## 页面布局

### 聊天页面布局

```
┌─────────────────────────────────────────────────────────────┐
│                      Header (64px)                          │
│  ┌────────┐  ┌──────────────────────┐  ┌─────────────────┐  │
│  │  Logo  │  │   Kiki Agent         │  │  🌙 / 👤        │  │
│  └────────┘  └──────────────────────┘  └─────────────────┘  │
├──────────┬──────────────────────────────────────────────────┤
│          │                                                   │
│  Sidebar │              Message List                       │
│  (280px) │          (max-width: 800px)                      │
│          │                                                   │
│          │  ┌─────────────────────────────────────────┐    │
│          │  │  🤖                                         │    │
│          │  │  ┌───────────────────────────────────┐   │    │
│          │  │  │  玻璃态卡片消息                      │   │    │
│          │  │  └───────────────────────────────────┘   │    │
│          │  │                                         │    │
│          │  │         Messages Scroll Area            │    │
│          │  │                                         │    │
│          │  └─────────────────────────────────────────┘    │
│          │                                                   │
│          │  ┌─────────────────────────────────────────┐    │
│          │  │  🪟 玻璃态输入框                          │    │
│          │  └─────────────────────────────────────────┘    │
│          │                                                   │
└──────────┴──────────────────────────────────────────────────┘
```

---

## 快速开始

### 安装依赖

```bash
cd frontend
npm install
```

### 开发模式

```bash
npm run dev
```

### 构建

```bash
npm run build
```

---

## 设计对比总结

| 特性 | WeKnora | Kiki |
|-----|---------|------|
| **主色** | 绿色 `#07c05f` | 琥珀橙 `#f59e0b` |
| **辅助色** | 无 | 靛蓝 `#6366f1` |
| **背景** | 纯色 `#eee` | 流动渐变 + 噪点 |
| **用户消息** | 浅绿纯色 `#8CE97F` | 琥珀橙渐变 |
| **卡片** | 纯色 + 边框 | 玻璃态 + 模糊 |
| **字体** | PingFang SC | Geist + IBM Plex Sans |
| **圆角** | 6px | 12px |
| **动画** | 基础 | 弹性 + 微交互 |
| **主题名** | 企业知识库 | 流动的智能 |

---

## 参考资料

- [React 官方文档](https://react.dev)
- [Vite 官方文档](https://vitejs.dev)
- [Zustand 文档](https://zustand-demo.pmnd.rs)
- [Framer Motion 文档](https://www.framer.com/motion/)
