# Kiki API 重构计划：Session/Message 独立接口

## 问题分析

### 当前问题
1. **agents.py 职责过重**：混合了 Agent CRUD、多 Agent 系统、聊天、会话验证等多种职责
2. **缺少独立的 Session 接口**：会话管理功能分散在多处
3. **缺少独立的 Message 接口**：消息管理功能不完整
4. **与 WeKnora99 结构不一致**：不利于团队协作和代码迁移

### 参考架构 (WeKnora99)
```
Handler Layer     →  session.go, message.go, chat.go
Service Layer     →  session.go, message.go
Repository Layer  →  session.go, message.go
```

## 重构方案

### 阶段一：创建 Session API

**新建文件：`app/api/v1/sessions.py`**

```python
# Session CRUD 接口
POST   /api/v1/sessions          # 创建会话
GET    /api/v1/sessions          # 获取会话列表（分页）
GET    /api/v1/sessions/{id}     # 获取会话详情
PATCH  /api/v1/sessions/{id}     # 更新会话（标题、描述、配置）
DELETE /api/v1/sessions/{id}     # 删除会话
POST   /api/v1/sessions/{id}/generate_title  # 自动生成标题
```

**职责：**
- 会话生命周期管理
- 会话配置管理（agent_config、context_config）
- 会话列表查询和分页

### 阶段二：创建 Message API

**新建文件：`app/api/v1/messages.py`**

```python
# Message 管理接口
GET    /api/v1/messages?session_id=xxx     # 获取消息列表（分页）
GET    /api/v1/messages/{id}               # 获取消息详情
PATCH  /api/v1/messages/{id}               # 编辑消息
DELETE /api/v1/messages/{id}               # 删除消息
GET    /api/v1/messages/search?q=xxx       # 消息搜索
```

**职责：**
- 消息历史管理
- 消息编辑和删除
- 消息搜索功能

### 阶段三：重构 Agents API

**修改文件：`app/api/v1/agents.py`**

**移除内容：**
- ❌ 会话验证逻辑 (`validate_session_access`) → 移至 `sessions.py`
- ❌ 消息持久化调用 → 移至 `messages.py` 或 `chat.py`
- ❌ 聊天接口 (`/chat`) → 保留在 `chat.py`

**保留内容：**
- ✅ Agent CRUD (`/agents`)
- ✅ 多 Agent 系统管理 (`/agents/router`, `/agents/supervisor`, `/agents/swarm`)
- ✅ Agent 统计信息 (`/agents/stats`)
- ✅ Agent 执行历史 (`/agents/executions`)

### 阶段四：更新 Chat API

**修改文件：`app/api/v1/chat.py`**

**新增：**
- 创建会话时自动调用 Session API
- 获取历史时调用 Message API

**保持：**
- 核心聊天接口 (`POST /`, `POST /stream`)
- 上下文管理接口

## Schema 设计

### Session Schemas (`app/schemas/session.py`)

```python
class SessionCreate(BaseModel):
    name: str
    agent_id: int | None = None
    agent_config: dict | None = None
    context_config: dict | None = None

class SessionUpdate(BaseModel):
    name: str | None = None
    agent_id: int | None = None
    agent_config: dict | None = None
    context_config: dict | None = None

class SessionResponse(BaseModel):
    id: str
    name: str
    user_id: int | None
    tenant_id: int | None
    agent_id: int | None
    message_count: int
    created_at: datetime
    updated_at: datetime

class SessionListResponse(BaseModel):
    items: list[SessionResponse]
    total: int
    page: int
    size: int
```

### Message Schemas (`app/schemas/message.py`)

```python
class MessageResponse(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    is_completed: bool
    knowledge_references: dict | None = None
    agent_steps: dict | None = None
    created_at: datetime

class MessageUpdate(BaseModel):
    content: str

class MessageListResponse(BaseModel):
    items: list[MessageResponse]
    total: int
    page: int
    size: int
```

## 实现顺序

### Step 1: 创建 Schemas
- [ ] `app/schemas/session.py`
- [ ] `app/schemas/message.py`

### Step 2: 创建 Services
- [ ] `app/services/session_service.py` - 封装业务逻辑
- [ ] `app/services/message_service.py` - 封装业务逻辑

### Step 3: 创建 APIs
- [ ] `app/api/v1/sessions.py`
- [ ] `app/api/v1/messages.py`

### Step 4: 更新路由注册
- [ ] `app/api/v1/__init__.py` - 注册新路由

### Step 5: 重构 Agents API
- [ ] 移除 `validate_session_access`（移至 sessions service）
- [ ] 移除聊天相关代码
- [ ] 清理导入

### Step 6: 更新 Chat API
- [ ] 使用新的 Session/Message services

## 关键文件清单

### 新建文件
| 文件 | 说明 |
|------|------|
| `app/schemas/session.py` | Session 数据模式 |
| `app/schemas/message.py` | Message 数据模式 |
| `app/services/session_service.py` | Session 业务逻辑 |
| `app/services/message_service.py` | Message 业务逻辑 |
| `app/api/v1/sessions.py` | Session API 路由 |
| `app/api/v1/messages.py` | Message API 路由 |

### 修改文件
| 文件 | 修改内容 |
|------|---------|
| `app/api/v1/__init__.py` | 注册新路由 |
| `app/api/v1/agents.py` | 移除会话/消息相关代码 |
| `app/api/v1/chat.py` | 使用新的 services |

## 用户确认的设计决策

1. **Thread 集成**：创建 Session 时自动创建对应的 Thread
2. **会话创建流程**：要求客户端先调用 Session API 创建会话（更规范）
3. **消息编辑**：支持编辑内容 + 重新生成 AI 后续消息

## 实现调整

### Session 创建流程
```python
POST /api/v1/sessions
  → 创建 ChatSession 记录
  → 创建 Thread 记录（用于 LangGraph）
  → 返回 session_id
```

### Chat 接口变更
```python
POST /api/v1/chat
  → 验证 session_id 是否存在
  → 不再自动创建会话
  → 返回 404 如果会话不存在
```

### Message 编辑 + 重新生成
```python
PATCH /api/v1/messages/{id}
  → 更新消息内容
  → 可选：regenerate=true 触发重新生成
  → 删除该消息之后的所有消息
  → 调用 Agent 重新生成回复
```

## 向后兼容性

- ⚠️ `POST /chat` 接口行为变更：不再自动创建会话
- ✅ 原有 `GET /chat/history/{session_id}` 接口保持不变
- ✅ 原有 Agent 接口保持不变

## 测试计划

1. **单元测试**
   - Session Service CRUD
   - Message Service CRUD

2. **集成测试**
   - Session API 端到端
   - Message API 端到端
   - Chat 与 Session/Message 集成

3. **手动测试**
   - 创建会话 → 发送消息 → 查看历史 → 删除会话
   - 编辑消息 → 重新生成
   - 搜索消息
