# Kiki Agent Framework - E2E 测试方案

> 文档版本: v1.0
> 创建日期: 2025-01-30
> 状态: 设计阶段

---

## 一、项目现状分析

### 1.1 技术栈

| 层级 | 技术选型 |
|------|----------|
| **后端** | FastAPI + Uvicorn + LangGraph + LangChain |
| **前端** | React 19 + Vite + TypeScript + Tailwind CSS |
| **数据库** | PostgreSQL (asyncpg) + SQLModel |
| **缓存** | Redis |
| **状态管理** | React Query (TanStack Query) |
| **认证** | JWT (python-jose + passlib) |
| **测试框架** | pytest + pytest-asyncio |

### 1.2 现有测试覆盖

```
tests/
├── conftest.py                    # pytest fixtures
├── unit/                          # 单元测试 (19 个)
├── integration/                   # 集成测试 (4 个)
│   ├── test_api.py               # 基础 API 测试
│   ├── test_agents_api.py        # Agent API 测试
│   ├── test_tools_api.py         # 工具 API 测试
│   └── test_multi_agent_e2e.py   # 多 Agent E2E 测试 (后端)
```

### 1.3 测试覆盖缺口

| 测试类型 | 覆盖率 | 缺口 |
|----------|--------|------|
| 单元测试 | ✅ 良好 | 部分业务逻辑待补充 |
| 集成测试 | ✅ 良好 | 后端 API 基本覆盖 |
| **前端 E2E** | ❌ **缺失** | **0% 覆盖** |
| 真实浏览器测试 | ❌ 缺失 | 未使用 Playwright/Cypress |
| 跨浏览器测试 | ❌ 缺失 | 未配置多浏览器 |
| 视觉回归测试 | ❌ 缺失 | 无视觉测试 |
| 性能测试 | ❌ 缺失 | 无性能基准测试 |

---

## 二、E2E 测试策略设计

### 2.1 测试金字塔

```
                /\
               /E2E\          ← 5-10 个关键用户流程
              /─────\
             /集成\          ← 50+ API/组件交互测试
            /────────\
           /单元测试\        ← 200+ 函数级测试
          /────────────\
```

### 2.2 E2E 测试原则

- 仅覆盖 **关键用户旅程**
- 使用 **真实浏览器** (Playwright)
- **Mock LLM API** (避免成本和延迟)
- **并行执行** 以提高速度
- 测试用户行为，不测试实现细节

### 2.3 关键测试场景

| 优先级 | 场景 | 描述 |
|--------|------|------|
| P0 | 用户认证流程 | 注册 → 登录 → Token 刷新 → 登出 |
| P0 | 基础聊天流程 | 发送消息 → 流式响应 → 历史记录 |
| P0 | Agent 创建与对话 | 创建 Router Agent → 发送消息 → 验证响应 |
| P1 | 多 Agent 协作 | Supervisor/Router 模式协作 |
| P1 | 工具调用 | 发送需要工具的请求 → 验证执行结果 |
| P1 | 会话管理 | 创建/删除会话 → 切换会话 |
| P2 | 错误处理 | 网络错误 → 显示友好提示 |
| P2 | 限流场景 | 超过速率限制 → 显示限流提示 |

---

## 三、Playwright 实施方案

### 3.1 目录结构

```
tests/
├── e2e/                           # 新增 E2E 测试目录
│   ├── playwright.config.ts       # Playwright 配置
│   ├── conftest.ts                # 全局 fixtures
│   ├── fixtures/                  # 测试数据 fixtures
│   │   ├── auth.ts                # 认证相关
│   │   ├── agents.ts              # Agent 相关
│   │   └── mock-server.ts         # Mock 服务器配置
│   ├── pages/                     # Page Object Model
│   │   ├── base.ts                # 基础页面类
│   │   ├── login.ts               # 登录页
│   │   ├── chat.ts                # 聊天页
│   │   └── agents.ts              # Agent 配置页
│   ├── helpers/                   # 辅助函数
│   │   ├── api-helpers.ts         # API 调用辅助
│   │   └── llm-mock.ts            # LLM Mock 响应
│   └── specs/                     # 测试规范
│       ├── auth.spec.ts           # 认证流程测试
│       ├── chat.spec.ts           # 聊天功能测试
│       ├── agents.spec.ts         # Agent 功能测试
│       └── tools.spec.ts          # 工具调用测试
```

### 3.2 Playwright 配置

```typescript
// tests/e2e/playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './specs',
  timeout: 30000,
  expect: { timeout: 5000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,

  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['junit', { outputFile: 'test-results/junit.xml' }],
    ['list']
  ],

  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],

  // 启动开发服务器
  webServer: {
    command: 'cd frontend && npm run dev',
    port: 5173,
    reuseExistingServer: !process.env.CI,
  },
});
```

### 3.3 Page Object Model 示例

```typescript
// tests/e2e/pages/chat.ts
import { Page, Locator, expect } from '@playwright/test';

export class ChatPage {
  readonly page: Page;
  readonly messageInput: Locator;
  readonly sendButton: Locator;
  readonly chatMessages: Locator;
  readonly typingIndicator: Locator;
  readonly newChatButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.messageInput = page.getByTestId('chat-input');
    this.sendButton = page.getByTestId('send-button');
    this.chatMessages = page.getByTestId('chat-message');
    this.typingIndicator = page.getByTestId('typing-indicator');
    this.newChatButton = page.getByTestId('new-chat-button');
  }

  async goto() {
    await this.page.goto('/');
  }

  async sendMessage(message: string) {
    await this.messageInput.fill(message);
    await this.sendButton.click();
  }

  async waitForResponse() {
    await expect(this.typingIndicator).toBeVisible({ timeout: 5000 });
    await expect(this.typingIndicator).toBeHidden({ timeout: 30000 });
  }

  async getLastMessage(): Promise<string> {
    const messages = this.chatMessages.all();
    const lastMessage = (await messages).at(-1);
    return (await lastMessage?.textContent()) || '';
  }

  async getMessageCount(): Promise<number> {
    return await this.chatMessages.count();
  }
}
```

### 3.4 LLM Mock 策略

```typescript
// tests/e2e/helpers/llm-mock.ts
export const mockLLMResponses = {
  simpleGreeting: {
    content: '你好！我是 Kiki 助手，有什么可以帮助你的？',
    role: 'assistant'
  },

  calculation: {
    content: '2 + 2 = 4',
    role: 'assistant',
    tool_calls: []
  },

  agentHandoff: {
    content: '让我帮你转接到销售专家。',
    role: 'assistant',
    metadata: { handoff_to: 'sales_agent' }
  }
};

// 设置路由拦截
export async function setupLLMMock(page: Page) {
  await page.route('**/api/v1/chat**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        content: mockLLMResponses.simpleGreeting.content,
        session_id: 'test-session',
        message_id: 'msg-123'
      })
    });
  });
}
```

### 3.5 测试用例示例

```typescript
// tests/e2e/specs/chat.spec.ts
import { test, expect } from '@playwright/test';
import { ChatPage } from '../pages/chat';
import { loginTestUser } from '../fixtures/auth';

test.describe('聊天功能', () => {
  let chatPage: ChatPage;

  test.beforeEach(async ({ page }) => {
    chatPage = new ChatPage(page);
    await loginTestUser(page);
    await chatPage.goto();
  });

  test('应该能发送和接收消息', async ({ page }) => {
    // 模拟 LLM 响应
    await page.route('**/api/v1/chat/stream', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'content-type': 'text/event-stream' },
        body: `data: {"content": "测试响应", "done": true}\n\n`
      });
    });

    const initialCount = await chatPage.getMessageCount();

    await chatPage.sendMessage('你好');
    await chatPage.waitForResponse();

    const finalCount = await chatPage.getMessageCount();
    expect(finalCount).toBe(initialCount + 2); // 用户消息 + AI 响应
  });

  test('应该能创建新对话', async ({ page }) => {
    await chatPage.sendMessage('第一条消息');
    await chatPage.waitForResponse();

    await chatPage.newChatButton.click();

    const messageCount = await chatPage.getMessageCount();
    expect(messageCount).toBe(0);
  });

  test('应该能查看聊天历史', async ({ page }) => {
    await chatPage.sendMessage('历史消息 1');
    await chatPage.waitForResponse();

    await page.reload();

    const messages = page.getByTestId('chat-message');
    await expect(messages).toHaveCount(2); // 用户 + AI
  });
});
```

---

## 四、关键 API 端点

### 4.1 认证相关

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/auth/register` | POST | 用户注册 |
| `/api/v1/auth/login` | POST | 用户登录 |
| `/api/v1/auth/me` | GET | 获取当前用户 |
| `/api/v1/auth/refresh` | POST | 刷新 Token |
| `/api/v1/auth/logout` | POST | 用户登出 |
| `/api/v1/auth/sessions` | GET | 获取会话列表 |
| `/api/v1/auth/sessions/{id}` | DELETE | 删除会话 |

### 4.2 聊天功能

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/chat` | POST | 发送消息 (同步) |
| `/api/v1/chat/stream` | POST | 发送消息 (流式 SSE) |
| `/api/v1/chat/history/{session_id}` | GET | 获取聊天历史 |
| `/api/v1/chat/history/{session_id}` | DELETE | 清空聊天历史 |
| `/api/v1/chat/context/{session_id}/stats` | GET | 获取上下文统计 |
| `/api/v1/chat/context/{session_id}` | DELETE | 清空上下文 |

### 4.3 Agent 系统

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/agents/router` | POST | 创建路由 Agent |
| `/api/v1/agents/router/{id}/chat` | POST | 路由 Agent 聊天 |
| `/api/v1/agents/supervisor` | POST | 创建监督 Agent |
| `/api/v1/agents/supervisor/{id}/chat` | POST | 监督 Agent 聊天 |
| `/api/v1/agents/swarm` | POST | 创建 Swarm Agent |
| `/api/v1/agents/swarm/{id}/chat` | POST | Swarm Agent 聊天 |
| `/api/v1/agents/systems` | GET | 获取所有 Agent 系统 |
| `/api/v1/agents/systems/{id}` | DELETE | 删除 Agent 系统 |

### 4.4 工具管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/tools` | GET | 获取工具列表 |
| `/api/v1/tools/{tool_name}` | GET | 获取工具详情 |

---

## 五、Mock 策略

### 5.1 为什么需要 Mock LLM？

| 原因 | 说明 |
|------|------|
| **成本** | 避免 E2E 测试消耗 API 配额 |
| **速度** | Mock 响应更快，测试更稳定 |
| **可控** | 确保测试结果可预测 |
| **离线** | CI/CD 环境无需 API Key |

### 5.2 Mock 实现方式

**方式一：后端 Mock 节点**
```python
# 在测试环境中使用 Mock LLM
if settings.ENV == "testing":
    llm_provider = MockLLMProvider()
```

**方式二：Playwright 路由拦截**
```typescript
await page.route('**/api/v1/**', async (route) => {
  // 返回预设响应
});
```

---

## 六、CI/CD 集成

```yaml
# .github/workflows/e2e.yml
name: E2E Tests

on:
  pull_request:
  push:
    branches: [main]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install Backend Dependencies
        run: uv sync --dev

      - name: Start Backend
        run: uv run uvicorn app.main:app &

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'

      - name: Install Frontend Dependencies
        run: cd frontend && npm ci

      - name: Install Playwright Browsers
        run: npx playwright install --with-deps

      - name: Run E2E Tests
        run: cd frontend && npm run test:e2e

      - name: Upload Test Report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: frontend/playwright-report/
```

---

## 七、实施步骤

### Phase 1: 基础设施 (Week 1)

```bash
# 1. 安装 Playwright
cd frontend
npm install -D @playwright/test
npx playwright install

# 2. 创建目录结构
mkdir -p tests/e2e/{fixtures,pages,helpers,specs}

# 3. 添加测试脚本
```

### Phase 2: 核心场景 (Week 2)

- [ ] 认证流程 E2E
- [ ] 基础聊天 E2E
- [ ] Agent 创建与对话 E2E

### Phase 3: 高级场景 (Week 3)

- [ ] 多 Agent 协作
- [ ] 工具调用验证
- [ ] 错误处理场景

### Phase 4: CI/CD 集成 (Week 4)

- [ ] GitHub Actions 配置
- [ ] 并行执行优化
- [ ] 测试报告集成

---

## 八、测试覆盖率目标

| 指标 | 目标值 | 当前值 |
|------|--------|--------|
| E2E 测试数量 | 20+ | 0 |
| 关键流程覆盖 | 100% | 0% |
| 测试执行时间 | <5 分钟 | N/A |
| 测试稳定性 | >95% | N/A |
| 测试维护成本 | 低 | N/A |

---

## 九、最佳实践

### 9.1 选择器策略

```typescript
// ❌ 不推荐: 脆弱的 CSS 选择器
page.get('.btn.btn-primary.submit-button').click();
page.get('div > form > div:nth-child(2) > input').type('text');

// ✅ 推荐: 语义化选择器
page.getByRole('button', { name: 'Submit' }).click();
page.getByLabel('Email address').type('user@example.com');
page.get('[data-testid="email-input"]').type('user@example.com');
```

### 9.2 等待策略

```typescript
// ❌ 不推荐: 固定延迟
await page.waitForTimeout(3000); // 不稳定！

// ✅ 推荐: 等待特定条件
await page.waitForLoadState('networkidle');
await expect(page.getByText('Welcome')).toBeVisible();
```

### 9.3 测试隔离

```typescript
// 每个测试独立运行，不依赖其他测试
test.beforeEach(async ({ page }) => {
  // 清理测试数据
  await cleanupTestData();
  // 设置测试环境
  await setupTestEnvironment();
});
```

---

## 十、参考资源

- [Playwright 官方文档](https://playwright.dev)
- [Testing Library 最佳实践](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)
- [E2E Testing Patterns](https://kentcdodds.com/blog/write-tests)
- [项目 E2E Testing Skill](../.claude/skills/e2e-testing-patterns/)

---

## 更新日志

| 日期 | 内容 | 操作者 |
|------|------|--------|
| 2025-01-30 | 创建 E2E 测试方案文档 | Claude |
