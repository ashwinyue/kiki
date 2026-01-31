# Kiki Agent Framework - API 文档

> 版本: v0.1.0
> 基础路径: `/api/v1`

---

## 目录

- [认证接口](#认证接口)
- [聊天接口](#聊天接口)
- [Agent 接口](#agent-接口)
- [工具接口](#工具接口)
- [评估接口](#评估接口)
- [错误响应](#错误响应)

---

## 认证接口

### 1. 用户注册

**端点:** `POST /auth/register`

**请求体:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "username": "johndoe"
}
```

**响应 (201):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "username": "johndoe"
  }
}
```

### 2. 用户登录

**端点:** `POST /auth/login`

**请求体:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**响应 (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 3. 刷新 Token

**端点:** `POST /auth/refresh`

**请求体:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**响应 (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 4. 获取当前用户

**端点:** `GET /auth/me`

**请求头:**
```
Authorization: Bearer <access_token>
```

**响应 (200):**
```json
{
  "id": "user_123",
  "email": "user@example.com",
  "username": "johndoe",
  "created_at": "2025-01-30T00:00:00Z"
}
```

### 5. 用户登出

**端点:** `POST /auth/logout`

**请求头:**
```
Authorization: Bearer <access_token>
```

**响应 (200):**
```json
{
  "message": "登出成功"
}
```

---

## 聊天接口

### 1. 发送消息（同步）

**端点:** `POST /chat`

**请求体:**
```json
{
  "message": "你好，请介绍一下自己",
  "session_id": "session_abc123",
  "user_id": "user_123"
}
```

**响应 (200):**
```json
{
  "content": "你好！我是 Kiki 助手，可以帮助你解答问题和完成各种任务...",
  "session_id": "session_abc123"
}
```

### 2. 发送消息（流式 SSE）

**端点:** `POST /chat/stream`

**请求体:**
```json
{
  "message": "写一首关于春天的诗",
  "session_id": "session_abc123",
  "user_id": "user_123",
  "stream_mode": "messages"
}
```

**流式模式:**
- `messages` - 令牌级流式输出（推荐）
- `updates` - 状态更新流
- `values` - 完整状态流

**响应 (text/event-stream):**
```
event: token
data: {"content": "春", "session_id": "session_abc123", "metadata": {...}}

event: token
data: {"content": "天", "session_id": "session_abc123", "metadata": {...}}

event: token
data: {"content": "来", "session_id": "session_abc123", "metadata": {...}}

event: done
data: {"session_id": "session_abc123", "done": true}
```

**客户端示例 (Python):**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat/stream",
    json={
        "message": "你好",
        "session_id": "test-123",
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

**客户端示例 (JavaScript):**
```javascript
const response = await fetch('/api/v1/chat/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message: '你好',
    session_id: 'session-123'
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  console.log(chunk);
}
```

### 3. 获取聊天历史

**端点:** `GET /chat/history/{session_id}`

**响应 (200):**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "你好"
    },
    {
      "role": "assistant",
      "content": "你好！我是 Kiki 助手..."
    }
  ],
  "session_id": "session_abc123"
}
```

### 4. 清空聊天历史

**端点:** `DELETE /chat/history/{session_id}`

**响应 (200):**
```json
{
  "status": "success",
  "message": "聊天历史已清除"
}
```

### 5. 获取上下文统计

**端点:** `GET /chat/context/{session_id}/stats`

**响应 (200):**
```json
{
  "session_id": "session_abc123",
  "message_count": 15,
  "token_estimate": 2400,
  "role_distribution": {
    "user": 7,
    "assistant": 7,
    "system": 1
  },
  "exists": true
}
```

### 6. 清空上下文

**端点:** `DELETE /chat/context/{session_id}`

**响应 (200):**
```json
{
  "status": "success",
  "message": "会话上下文已清除"
}
```

---

## Agent 接口

多 Agent 系统 API 支持 Router、Supervisor、Swarm 三种协作模式。

### 管理接口

#### 列出所有 Agent 系统

**端点:** `GET /agents/systems`

**响应 (200):**
```json
[
  {
    "id": "router_abc123",
    "type": "router",
    "name": "客服路由系统",
    "agents": ["Sales", "Support"]
  },
  {
    "id": "supervisor_xyz789",
    "type": "supervisor",
    "name": "内容生产系统",
    "agents": ["Researcher", "Writer", "Reviewer"]
  }
]
```

#### 获取单个 Agent 系统

**端点:** `GET /agents/systems/{system_id}`

**响应 (200):**
```json
{
  "id": "router_abc123",
  "type": "router",
  "name": "客服路由系统",
  "agents": ["Sales", "Support"]
}
```

#### 删除 Agent 系统

**端点:** `DELETE /agents/systems/{system_id}`

**响应 (200):**
```json
{
  "deleted": true
}
```

---

### Router Agent - 路由模式

路由 Agent 根据用户意图将请求分发到不同的子 Agent。

#### 创建路由 Agent

**端点:** `POST /agents/router`

**请求体:**
```json
{
  "name": "客服路由系统",
  "agents": [
    {
      "name": "Sales",
      "system_prompt": "你是销售专家，负责产品推荐和价格咨询",
      "tools": ["search_products", "calculate_discount"]
    },
    {
      "name": "Support",
      "system_prompt": "你是客服专家，负责订单查询和售后处理",
      "tools": ["check_order_status", "process_refund"]
    }
  ],
  "router_prompt": "根据用户意图选择 Sales(销售) 或 Support(客服)"
}
```

**响应 (200):**
```json
{
  "name": "客服路由系统",
  "type": "router",
  "agents": ["Sales", "Support"],
  "session_id": "router_abc123"
}
```

#### 路由 Agent 聊天

**端点:** `POST /agents/router/{system_id}/chat`

**请求参数 (Query):**
- `message` - 用户消息
- `session_id` - 会话 ID
- `user_id` - 用户 ID（可选）

**响应 (200):**
```json
{
  "content": "让我帮你查询产品信息...",
  "session_id": "session_123",
  "agent_name": "router_abc123"
}
```

---

### Supervisor Agent - 监督者模式

监督 Agent 协调多个 Worker Agent 完成复杂任务。

#### 创建监督 Agent

**端点:** `POST /agents/supervisor`

**请求体:**
```json
{
  "name": "内容生产系统",
  "workers": [
    {
      "name": "Researcher",
      "system_prompt": "你是研究员，负责收集和分析信息",
      "tools": ["search", "crawl"]
    },
    {
      "name": "Writer",
      "system_prompt": "你是写手，负责整理和撰写报告",
      "tools": []
    },
    {
      "name": "Reviewer",
      "system_prompt": "你是审核员，负责审核报告质量",
      "tools": []
    }
  ],
  "supervisor_prompt": "协调 Researcher、Writer、Reviewer 完成报告撰写任务"
}
```

**响应 (200):**
```json
{
  "name": "内容生产系统",
  "type": "supervisor",
  "agents": ["Researcher", "Writer", "Reviewer"],
  "session_id": "supervisor_xyz789"
}
```

#### 监督 Agent 聊天

**端点:** `POST /agents/supervisor/{system_id}/chat`

**请求参数 (Query):**
- `message` - 用户消息
- `session_id` - 会话 ID
- `user_id` - 用户 ID（可选）

**响应 (200):**
```json
{
  "content": "已完成报告撰写，经过审核后交付。",
  "session_id": "session_123",
  "agent_name": "supervisor_xyz789"
}
```

---

### Swarm Agent - 群体模式

Swarm Agent 支持多个 Agent 之间动态切换控制权（Handoff）。

#### 创建 Swarm Agent

**端点:** `POST /agents/swarm`

**请求体:**
```json
{
  "name": "客服协作系统",
  "agents": [
    {
      "name": "Alice",
      "system_prompt": "你是 Alice，销售专家。遇到技术问题时转给 Bob。",
      "tools": ["search_products", "calculate_discount"]
    },
    {
      "name": "Bob",
      "system_prompt": "你是 Bob，客服专家。遇到销售咨询时转给 Alice。",
      "tools": ["check_order_status", "process_refund"]
    }
  ],
  "handoff_mapping": {
    "Alice": ["Bob"],
    "Bob": ["Alice"]
  },
  "default_agent": "Alice"
}
```

**字段说明：**
- `handoff_mapping` - 指定每个 Agent 可以切换到的目标列表
- `default_agent` - 默认激活的 Agent

**响应 (200):**
```json
{
  "name": "客服协作系统",
  "type": "swarm",
  "agents": ["Alice", "Bob"],
  "session_id": "swarm_def456"
}
```

#### Swarm Agent 聊天

**端点:** `POST /agents/swarm/{system_id}/chat`

**请求参数 (Query):**
- `message` - 用户消息
- `session_id` - 会话 ID
- `user_id` - 用户 ID（可选）

**响应 (200):**
```json
{
  "content": "我是 Bob，让我帮你处理退款问题...",
  "session_id": "session_123",
  "agent_name": "swarm_def456"
}
```

---

### 使用示例

#### Python 客户端

```python
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

# 1. 创建路由 Agent
router_data = {
    "name": "客服路由系统",
    "agents": [
        {"name": "Sales", "system_prompt": "你是销售专家", "tools": []},
        {"name": "Support", "system_prompt": "你是客服专家", "tools": []}
    ]
}

response = requests.post(f"{BASE_URL}/agents/router", json=router_data)
result = response.json()
system_id = result["session_id"]

# 2. 使用路由 Agent 聊天
chat_response = requests.post(
    f"{BASE_URL}/agents/router/{system_id}/chat",
    params={
        "message": "我想买一个手机",
        "session_id": "user-123"
    }
)

print(chat_response.json()["content"])

# 3. 列出所有系统
systems = requests.get(f"{BASE_URL}/agents/systems").json()
print(json.dumps(systems, indent=2, ensure_ascii=False))
```

#### JavaScript 客户端

```javascript
const BASE_URL = 'http://localhost:8000/api/v1';

// 1. 创建路由 Agent
const routerData = {
  name: '客服路由系统',
  agents: [
    { name: 'Sales', system_prompt: '你是销售专家', tools: [] },
    { name: 'Support', system_prompt: '你是客服专家', tools: [] }
  ]
};

const createResponse = await fetch(`${BASE_URL}/agents/router`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(routerData)
});

const { session_id } = await createResponse.json();

// 2. 使用路由 Agent 聊天
const chatResponse = await fetch(
  `${BASE_URL}/agents/router/${session_id}/chat?message=我想买一个手机&session_id=user-123`,
  { method: 'POST' }
);

const { content } = await chatResponse.json();
console.log(content);
```

---

## 工具接口

### 1. 获取工具列表

**端点:** `GET /tools`

**响应 (200):**
```json
{
  "tools": [
    {
      "name": "get_weather",
      "description": "获取指定城市的天气信息",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "城市名称"
          }
        },
        "required": ["city"]
      }
    },
    {
      "name": "calculator",
      "description": "执行数学计算",
      "parameters": {
        "type": "object",
        "properties": {
          "expression": {
            "type": "string",
            "description": "数学表达式"
          }
        },
        "required": ["expression"]
      }
    }
  ]
}
```

### 2. 获取工具详情

**端点:** `GET /tools/{tool_name}`

**响应 (200):**
```json
{
  "name": "get_weather",
  "description": "获取指定城市的天气信息",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "城市名称"
      }
    },
    "required": ["city"]
  },
  "examples": [
    {
      "input": {"city": "北京"},
      "output": "北京 今天晴朗，温度 25°C"
    }
  ]
}
```

---

## 评估接口

### 1. 创建评估任务

**端点:** `POST /evaluation`

**请求体:**
```json
{
  "dataset_id": "dataset_123",
  "agent_config": {
    "type": "chat",
    "system_prompt": "你是一个有用的助手"
  },
  "evaluators": ["accuracy", "relevance", "helpfulness"]
}
```

**响应 (200):**
```json
{
  "evaluation_id": "eval_abc123",
  "status": "pending",
  "created_at": "2025-01-30T00:00:00Z"
}
```

### 2. 获取评估结果

**端点:** `GET /evaluation/{evaluation_id}`

**响应 (200):**
```json
{
  "evaluation_id": "eval_abc123",
  "status": "completed",
  "results": {
    "accuracy": 0.92,
    "relevance": 0.88,
    "helpfulness": 0.95
  },
  "total_samples": 100,
  "passed_samples": 88
}
```

### 3. 获取评估报告

**端点:** `GET /evaluation/{evaluation_id}/report`

**响应 (200):**
```json
{
  "evaluation_id": "eval_abc123",
  "summary": {
    "overall_score": 0.91,
    "total_samples": 100,
    "passed_samples": 88
  },
  "details": [
    {
      "sample_id": "sample_1",
      "input": "问题内容",
      "expected_output": "期望答案",
      "actual_output": "实际答案",
      "scores": {
        "accuracy": 1.0,
        "relevance": 0.9
      }
    }
  ]
}
```

---

## 错误响应

### 错误响应格式

所有错误响应遵循统一格式：

```json
{
  "error": "错误类型",
  "detail": "用户友好的错误描述",
  "category": "错误类别（仅开发环境）"
}
```

### 错误类别

| HTTP 状态码 | 错误类别 | 说明 |
|-------------|----------|------|
| 400 | `validation` | 请求参数验证失败 |
| 401 | `auth` | 未认证或 Token 无效 |
| 403 | `permission` | 无权限访问 |
| 404 | `not_found` | 资源不存在 |
| 429 | `rate_limit` | 请求过于频繁 |
| 500 | `internal` | 服务器内部错误 |

### 错误示例

**验证错误 (400):**
```json
{
  "error": "Validation Error",
  "detail": "email: 邮箱格式无效",
  "category": "validation"
}
```

**认证错误 (401):**
```json
{
  "error": "Unauthorized",
  "detail": "Token 已过期，请重新登录",
  "category": "auth"
}
```

**限流错误 (429):**
```json
{
  "error": "Rate Limit Exceeded",
  "detail": "请求过于频繁，请稍后再试",
  "category": "rate_limit",
  "retry_after": 60
}
```

**内部错误 (500):**
```json
{
  "error": "Internal Server Error",
  "detail": "服务暂时不可用，请稍后再试"
}
```

---

## 请求头

### 标准请求头

| 请求头 | 必需 | 说明 |
|--------|------|------|
| `Content-Type` | 是 | `application/json` |
| `Authorization` | 条件 | Bearer Token（需要认证的接口） |
| `X-Request-ID` | 否 | 请求追踪 ID（自动生成） |

### 响应头

| 响应头 | 说明 |
|--------|------|
| `X-Request-ID` | 请求追踪 ID |
| `X-RateLimit-Remaining` | 剩余请求次数 |
| `X-RateLimit-Reset` | 限流重置时间 |

---

## 认证方式

Kiki API 使用 JWT Bearer Token 认证。

### 设置 Token

```bash
curl -H "Authorization: Bearer <your_token>" \
  http://localhost:8000/api/v1/chat/history/session_123
```

### Token 刷新流程

1. 使用 `refresh_token` 调用刷新接口获取新的 `access_token`
2. 当 `access_token` 过期时，应自动刷新
3. `refresh_token` 有效期为 7 天

---

## 分页

列表接口支持分页：

### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `page` | int | 1 | 页码 |
| `limit` | int | 20 | 每页数量 |
| `sort` | string | `-created_at` | 排序字段（`-` 表示降序） |

### 响应格式

```json
{
  "data": [...],
  "meta": {
    "total": 100,
    "page": 1,
    "limit": 20,
    "pages": 5
  }
}
```

---

## 版本控制

API 版本通过 URL 路径控制：

- 当前版本: `/api/v1/`
- 未来版本: `/api/v2/`

---

## SDK 和客户端

### Python SDK 示例

```python
from kiki import KikiClient

client = KikiClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# 同步聊天
response = client.chat.chat(
    message="你好",
    session_id="session-123"
)
print(response.content)

# 流式聊天
async for chunk in client.chat.stream(
    message="写一首诗",
    session_id="session-123"
):
    print(chunk.content, end="")
```

### JavaScript SDK 示例

```javascript
import { KikiClient } from '@kiki/sdk';

const client = new KikiClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// 同步聊天
const response = await client.chat.chat({
  message: '你好',
  sessionId: 'session-123'
});
console.log(response.content);

// 流式聊天
for await (const chunk of client.chat.stream({
  message: '写一首诗',
  sessionId: 'session-123'
})) {
  process.stdout.write(chunk.content);
}
```
