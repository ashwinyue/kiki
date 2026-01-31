# Kiki 前端 API 对接文档

> 版本: v1.0.0
> 更新日期: 2025-01-31
> 说明: 后端 API 与前端页面对接映射

---

## 目录

- [API 映射总览](#api-映射总览)
- [认证模块](#认证模块)
- [聊天模块](#聊天模块)
- [Agent 管理模块](#agent-管理模块)
- [工具管理模块](#工具管理模块)
- [API Key 管理](#api-key-管理)
- [租户管理模块](#租户管理模块)
- [MCP 服务管理](#mcp-服务管理)
- [评估模块](#评估模块)
- [数据类型定义](#数据类型定义)

---

## API 映射总览

### 页面与 API 对应关系

| 前端页面/组件 | 后端 API 端点 | 认证要求 |
|-------------|-------------|---------|
| **登录页** (`LoginPage`) | `POST /api/v1/auth/login` | 无 |
| **注册页** (`RegisterPage`) | `POST /api/v1/auth/register` | 无 |
| **聊天页面** (`ChatPage`) | `POST /api/v1/chat/stream` | 无 (session_id) |
| **会话列表** (`ChatSidebar`) | `GET /api/v1/auth/sessions` | Bearer Token |
| **创建会话** | `POST /api/v1/auth/sessions` | Bearer Token |
| **删除会话** | `DELETE /api/v1/auth/sessions/{id}` | Bearer Token |
| **Agent 管理页** (`AgentsPage`) | `GET /api/v1/agents/list` | Bearer Token |
| **创建 Agent** | `POST /api/v1/agents` | Bearer Token |
| **Agent 详情** | `GET /api/v1/agents/{id}` | Bearer Token |
| **工具列表** (`ToolsPage`) | `GET /api/v1/tools` | 无 |
| **设置页** (`SettingsPage`) | `GET /api/v1/auth/me` | Bearer Token |
| **API Keys** (`ApiKeysPage`) | `GET /api/v1/api-keys` | Bearer Token |
| **创建 API Key** | `POST /api/v1/api-keys` | Bearer Token |

---

## 认证模块

### 1. 用户登录

**前端位置**: `src/pages/auth/LoginPage.tsx`

**API 端点**: `POST /api/v1/auth/login`

**请求格式**:
```typescript
interface LoginRequest {
  username: string;  // 邮箱
  password: string;
}
```

**响应格式**:
```typescript
interface LoginResponse {
  access_token: string;
  token_type: 'bearer';
  expires_at: string;  // ISO 8601 时间格式
}
```

**交互流程**:
```
用户输入 → 表单验证 → API 调用 → 存储 Token → 跳转聊天页
```

### 2. 用户注册

**前端位置**: `src/pages/auth/RegisterPage.tsx`

**API 端点**: `POST /api/v1/auth/register`

**请求格式**:
```typescript
interface RegisterRequest {
  email: string;
  password: string;  // 8-100 字符
  full_name?: string;
}
```

**响应格式**:
```typescript
interface RegisterResponse {
  id: number;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  access_token: string;
  token_type: 'bearer';
}
```

### 3. 获取当前用户

**前端位置**: `src/stores/authStore.ts`

**API 端点**: `GET /api/v1/auth/me`

**请求头**:
```
Authorization: Bearer <token>
```

**响应格式**:
```typescript
interface User {
  id: number;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
}
```

---

## 聊天模块

### 1. 流式聊天

**前端位置**: `src/hooks/useChat.ts`

**API 端点**: `POST /api/v1/chat/stream`

**请求格式**:
```typescript
interface ChatStreamRequest {
  message: string;
  session_id: string;
  user_id?: string;
  stream_mode?: 'messages' | 'updates' | 'values';
}
```

**响应格式**: Server-Sent Events (SSE)
```typescript
interface ChatStreamEvent {
  event: 'token' | 'update' | 'state' | 'done';
  data: {
    content?: string;
    session_id: string;
    metadata?: Record<string, unknown>;
  };
}
```

**交互流程**:
```
输入消息 → 创建用户消息 → 发送请求 → 接收 SSE 流 → 逐字显示 → 完成
```

### 2. 获取聊天历史

**API 端点**: `GET /api/v1/chat/history/{session_id}`

**响应格式**:
```typescript
interface ChatHistoryResponse {
  messages: Array<{
    role: 'user' | 'assistant' | 'system';
    content: string;
  }>;
  session_id: string;
}
```

### 3. 清除历史

**API 端点**: `DELETE /api/v1/chat/history/{session_id}`

### 4. 上下文统计

**API 端点**: `GET /api/v1/chat/context/{session_id}/stats`

**响应格式**:
```typescript
interface ContextStatsResponse {
  session_id: string;
  message_count: number;
  token_estimate: number;
  role_distribution: {
    user: number;
    assistant: number;
  };
  exists: boolean;
}
```

---

## Agent 管理模块

### 1. Agent 列表

**前端位置**: `src/pages/agents/AgentsPage.tsx`

**API 端点**: `GET /api/v1/agents/list`

**请求参数**:
```typescript
interface AgentListQuery {
  agent_type?: 'single' | 'router' | 'supervisor' | 'worker' | 'handoff';
  status?: 'active' | 'disabled' | 'deleted';
  page?: number;
  size?: number;
}
```

**响应格式**:
```typescript
interface AgentListResponse {
  agents: Agent[];
  total: number;
  page: number;
  size: number;
}

interface Agent {
  id: string;
  name: string;
  description?: string;
  agent_type: string;
  model_name: string;
  system_prompt: string;
  temperature: number;
  max_tokens?: number;
  config?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}
```

### 2. 创建 Agent

**API 端点**: `POST /api/v1/agents`

**请求格式**:
```typescript
interface CreateAgentRequest {
  name: string;
  description?: string;
  agent_type: 'single' | 'router' | 'supervisor' | 'worker' | 'handoff';
  model_name: string;
  system_prompt: string;
  temperature: number;
  max_tokens?: number;
  config?: Record<string, unknown>;
}
```

### 3. Agent 统计

**API 端点**: `GET /api/v1/agents/stats`

**响应格式**:
```typescript
interface AgentStatsResponse {
  total: number;
  active: number;
  by_type: Record<string, number>;
}
```

### 4. 执行历史

**API 端点**: `GET /api/v1/agents/executions`

**请求参数**:
```typescript
interface ExecutionsQuery {
  agent_id?: string;
  limit?: number;
}
```

---

## 工具管理模块

### 1. 工具列表

**前端位置**: `src/pages/tools/ToolsPage.tsx`

**API 端点**: `GET /api/v1/tools`

**响应格式**:
```typescript
interface ToolsResponse {
  tools: Array<{
    name: string;
    description: string;
    args_schema: string;
  }>;
  count: number;
}
```

### 2. 工具详情

**API 端点**: `GET /api/v1/tools/{tool_name}`

---

## API Key 管理

### 1. API Key 列表

**前端位置**: `src/pages/settings/ApiKeysPage.tsx`

**API 端点**: `GET /api/v1/api-keys`

**请求参数**:
```typescript
interface ApiKeysQuery {
  key_type?: 'personal' | 'service' | 'mcp' | 'webhook';
  status?: 'active' | 'revoked' | 'expired';
}
```

**响应格式**:
```typescript
interface ApiKeysResponse {
  keys: Array<{
    id: number;
    name: string;
    key_prefix: string;
    key_type: string;
    status: string;
    scopes: string[];
    expires_at: string | null;
    created_at: string;
    last_used: string | null;
  }>;
}
```

### 2. 创建 API Key

**API 端点**: `POST /api/v1/api-keys`

**请求格式**:
```typescript
interface CreateApiKeyRequest {
  name: string;
  key_type: 'personal' | 'service' | 'mcp' | 'webhook';
  scopes: string[];
  expires_in_days?: number;
  description?: string;
  rate_limit?: number;
}
```

**响应格式**:
```typescript
interface CreateApiKeyResponse {
  id: number;
  name: string;
  key: string;  // 完整的 API Key（仅返回一次）
  key_prefix: string;
  key_type: string;
  status: string;
  scopes: string[];
  expires_at: string | null;
  created_at: string;
}
```

### 3. 删除/吊销 API Key

**API 端点**:
- `DELETE /api/v1/api-keys/{id}` - 删除
- `POST /api/v1/api-keys/{id}/revoke` - 吊销

### 4. API Key 统计

**API 端点**: `GET /api/v1/api-keys/stats/me`

**响应格式**:
```typescript
interface ApiKeyStatsResponse {
  user_id: number;
  total_keys: number;
  by_status: Record<string, number>;
  by_type: Record<string, number>;
}
```

---

## 租户管理模块

### 1. 租户列表

**前端位置**: `src/pages/admin/TenantsPage.tsx` (管理员)

**API 端点**: `GET /api/v1/tenants`

**请求参数**:
```typescript
interface TenantsQuery {
  status?: 'active' | 'suspended' | 'pending';
  keyword?: string;
  page?: number;
  size?: number;
}
```

### 2. 创建租户

**API 端点**: `POST /api/v1/tenants`

### 3. 租户配置

**API 端点**: `GET /api/v1/tenants/me/config`

**响应格式**:
```typescript
interface TenantConfigResponse {
  tenant_id: string;
  config: Record<string, unknown>;
}
```

---

## MCP 服务管理

### 1. MCP 服务列表

**前端位置**: `src/pages/mcp/McpServicesPage.tsx`

**API 端点**: `GET /api/v1/mcp-services`

**请求参数**:
```typescript
interface McpServicesQuery {
  include_disabled?: boolean;
}
```

**响应格式**:
```typescript
interface McpServicesResponse {
  services: Array<{
    id: string;
    name: string;
    description?: string;
    enabled: boolean;
    transport_type: 'stdio' | 'http' | 'sse';
    url?: string;
    created_at: string;
  }>;
}
```

### 2. 创建 MCP 服务

**API 端点**: `POST /api/v1/mcp-services`

**请求格式**:
```typescript
interface CreateMcpServiceRequest {
  name: string;
  description?: string;
  enabled?: boolean;
  transport_type: 'stdio' | 'http' | 'sse';
  url?: string;
  headers?: Record<string, string>;
  auth_config?: Record<string, unknown>;
  advanced_config?: Record<string, unknown>;
  stdio_config?: Record<string, unknown>;
  env_vars?: Record<string, string>;
}
```

---

## 评估模块

### 1. 运行评估

**前端位置**: `src/pages/evaluation/EvaluationPage.tsx`

**API 端点**: `POST /api/v1/evaluation/run`

**请求格式**:
```typescript
interface RunEvaluationRequest {
  dataset_name: string;
  evaluators: string[];
  agent_type: string;
  session_id_prefix: string;
  max_entries?: number;
  categories?: string[];
  stream?: boolean;
}
```

**响应格式**:
```typescript
interface RunEvaluationResponse {
  run_id: string;
  status: string;
  message: string;
}
```

### 2. 评估结果

**API 端点**: `GET /api/v1/evaluation/results/{run_id}`

**响应格式**:
```typescript
interface EvaluationResultsResponse {
  run_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  results?: Array<{
    question: string;
    answer: string;
    score: number;
    feedback: string;
  }>;
  summary?: {
    total: number;
    passed: number;
    failed: number;
    avg_score: number;
  };
}
```

### 3. 数据集列表

**API 端点**: `GET /api/v1/evaluation/datasets`

**响应格式**:
```typescript
interface DatasetsResponse {
  datasets: Array<{
    name: string;
    description: string;
    entry_count: number;
    version: string;
    categories: string[];
  }>;
}
```

---

## 数据类型定义

### 通用类型

```typescript
// src/types/common.ts
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  meta?: {
    total: number;
    page: number;
    limit: number;
  };
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}
```

### 认证类型

```typescript
// src/types/auth.ts
export interface User {
  id: number;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  created_at?: string;
}

export interface AuthTokens {
  access_token: string;
  token_type: 'bearer';
  expires_at: string;
}

export interface Session {
  id: string;
  name: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}
```

### 聊天类型

```typescript
// src/types/chat.ts
export type MessageRole = 'user' | 'assistant' | 'system';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface Session {
  id: string;
  name: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface StreamEvent {
  event: 'token' | 'update' | 'state' | 'done' | 'error';
  data: {
    content?: string;
    session_id: string;
    metadata?: Record<string, unknown>;
  };
}
```

### Agent 类型

```typescript
// src/types/agent.ts
export type AgentType = 'single' | 'router' | 'supervisor' | 'worker' | 'handoff';
export type AgentStatus = 'active' | 'disabled' | 'deleted';

export interface Agent {
  id: string;
  name: string;
  description?: string;
  agent_type: AgentType;
  model_name: string;
  system_prompt: string;
  temperature: number;
  max_tokens?: number;
  config?: Record<string, unknown>;
  status: AgentStatus;
  created_at: string;
  updated_at: string;
}

export interface AgentExecution {
  id: string;
  agent_id: string;
  status: string;
  input: string;
  output?: string;
  error?: string;
  started_at: string;
  completed_at?: string;
}
```

---

## 前端实现优先级

### P0 - 核心功能 (必须实现)

1. **认证流程**
   - [ ] 登录页面
   - [ ] 注册页面
   - [ ] Token 存储和刷新
   - [ ] 路由守卫

2. **聊天功能**
   - [ ] 聊天页面布局
   - [ ] SSE 流式消息接收
   - [ ] 消息列表渲染
   - [ ] 输入框组件
   - [ ] 会话创建和切换

### P1 - 重要功能

3. **Agent 管理**
   - [ ] Agent 列表页
   - [ ] Agent 创建表单
   - [ ] Agent 详情页
   - [ ] Agent 配置编辑

4. **设置页面**
   - [ ] 用户信息展示
   - [ ] API Key 管理
   - [ ] 主题切换

### P2 - 辅助功能

5. **工具管理**
   - [ ] 工具列表展示
   - [ ] 工具详情查看

6. **MCP 服务**
   - [ ] MCP 服务列表
   - [ ] MCP 服务配置

### P3 - 高级功能

7. **评估系统**
   - [ ] 评估运行界面
   - [ ] 结果展示页面

---

## API 错误处理

### 错误响应格式

```typescript
interface ApiError {
  detail: string;
  status_code: number;
  error_code?: string;
}
```

### 常见错误码

| 状态码 | 说明 | 前端处理 |
|-------|------|---------|
| 400 | 请求参数错误 | 表单验证提示 |
| 401 | 未认证 | 跳转登录页 |
| 403 | 权限不足 | 显示权限错误 |
| 404 | 资源不存在 | 显示 404 页面 |
| 422 | 验证失败 | 字段级错误提示 |
| 429 | 请求过于频繁 | 显示限流提示 |
| 500 | 服务器错误 | 显示友好错误页 |

---

## WebSocket 接口

### 连接地址

```
```

### 消息格式

**客户端发送**:
```typescript
interface WSChatMessage {
  action: 'chat';
  prompt: string;
  system?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  session_id?: string;
}
```

**服务端推送**:
```typescript
interface WSChatEvent {
  type: 'delta' | 'done' | 'error' | 'metadata' | 'thinking';
  content?: string;
  metadata?: Record<string, unknown>;
  timestamp: number;
}
```

---

## 总结

本文档提供了 Kiki 前端与后端 API 的完整对接映射。前端开发时请严格按照：

1. **API 端点路径**: 使用 `/api/v1` 前缀
2. **认证方式**: Bearer Token 或 API Key
3. **数据格式**: JSON 请求/响应
4. **错误处理**: 统一的错误响应格式
5. **类型安全**: 使用 TypeScript 类型定义
