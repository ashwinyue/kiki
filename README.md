# Kiki Agent Framework

> 企业级 Agent 开发脚手架 - 基于 FastAPI + LangGraph

## 特性

- **LangGraph 集成**: 开箱即用的 Agent 工作流编排
- **生产就绪**: 认证、授权、可观测性完整支持
- **模块化架构**: 清晰的分层设计，易于扩展
- **类型安全**: 完整的类型注解，支持 mypy 检查
- **企业级脚手架**: 专注于 Agent 开发，移除复杂的知识库管理

## 架构亮点

### ✅ 标准 LangGraph 模式

```python
# 工具在编译时绑定，节点无需重复获取
graph = compile_chat_graph(
    llm_service=llm_service,
    system_prompt=system_prompt,
    tenant_id=tenant_id
)
```

### ✅ 上下文窗口管理

```python
class ChatState(MessagesState):
    max_tokens: int = 8000      # 自动截断超长对话
    max_messages: int = 50     # 消息数量限制

    def trim_messages(self):
        # 滑动窗口保留最近消息
        ...
```

### ✅ PostgreSQL Checkpointer

- 状态持久化：自动保存对话历史
- 多租户隔离：每个 session_id 独立存储
- 故障恢复：服务重启后状态不丢失

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env

# 启动服务
uv run uvicorn app.main:app --reload
```

## 项目结构

```
kiki/
├── app/
│   ├── agent/          # Agent 核心（LangGraph）
│   │   ├── graph/      # 工作流图构建
│   │   ├── memory/     # 记忆系统
│   │   ├── tools/      # 工具注册和执行
│   │   └── streaming/  # 流式响应
│   ├── api/            # API 路由层
│   │   └── v1/         # v1 版本 API
│   ├── config/         # 配置管理
│   ├── llm/            # LLM 服务抽象
│   ├── middleware/     # 中间件（认证、限流等）
│   ├── observability/  # 可观测性（日志、指标、审计）
│   ├── auth/           # 认证授权
│   ├── models/         # 数据模型
│   ├── schemas/        # Pydantic 模式
│   ├── services/       # 业务服务
│   │   ├── core/       # 核心服务（session、auth等）
│   │   ├── agent/      # Agent 相关服务
│   │   ├── llm/        # LLM 相关服务
│   │   └── web/        # Web 搜索服务
│   ├── repositories/   # 数据访问
│   ├── infra/          # 基础设施（数据库、缓存、Redis）
│   ├── rate_limit/     # 限流
│   └── utils/          # 工具函数
├── tests/              # 测试
└── pyproject.toml      # 项目配置（使用 uv）
```

## 核心 API

### 聊天接口

```python
# 同步聊天
POST /api/v1/chat
{
  "message": "你好",
  "session_id": "session-123"
}

# 流式聊天（SSE）
POST /api/v1/chat/stream
{
  "message": "你好",
  "session_id": "session-123",
  "stream_mode": "messages"
}
```

### Agent 管理

- `POST /api/v1/agents` - 创建 Agent
- `GET /api/v1/agents` - 列出 Agents
- `GET /api/v1/agents/{id}` - 获取 Agent 详情

### 会话管理

- `GET /api/v1/sessions` - 列出会话
- `GET /api/v1/sessions/{id}` - 获取会话详情
- `DELETE /api/v1/sessions/{id}` - 删除会话
- `DELETE /api/v1/sessions/{id}/history` - 清除聊天历史

## 技术栈

- **Web**: FastAPI + Uvicorn
- **Agent**: LangGraph + LangChain
- **数据库**: PostgreSQL (asyncpg) + SQLModel
- **缓存**: Redis
- **可观测性**: structlog + Langfuse + Prometheus
- **认证**: JWT + API Key
- **限流**: slowapi + token bucket
- **测试**: pytest + pytest-asyncio + pytest-cov
- **代码质量**: ruff + mypy

## 开发

```bash
# 运行测试
uv run pytest

# 代码检查
uv run ruff check .
uv run mypy app/

# 启动开发服务器
uv run uvicorn app.main:app --reload
```

## 架构设计

### Agent 工作流

```
用户消息 → START → chat_node
                    ↓
              有工具调用？
                    ↓
             [tools_node] → 返回 chat_node
                    ↓
                 返回 → END
```

### 数据流

```
Request → API → Agent → LLM (带工具绑定)
                ↓
         PostgreSQL Checkpointer
                ↓
         Database Messages (备份)
```

## 文档

- [架构设计](AGENTS.md) - 企业级架构指南
- [API 文档](docs/api.md) - API 参考
- [开发指南](docs/development.md) - 开发规范

## 更新日志

### v0.2.0 (当前)

- ✅ 重构为标准 LangGraph 模式
- ✅ 添加上下文窗口管理
- ✅ 简化配置传递机制
- ✅ 移除冗余的 Context Manager
- ✅ 工具在编译时绑定，性能提升

### v0.1.0

- 初始版本
- Celery 任务队列
- 知识库管理

## 许可证

MIT
