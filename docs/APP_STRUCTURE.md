# App 模块架构文档

> Kiki Agent Framework 应用模块详细架构说明

## 目录结构总览

```
app/
├── main.py                    # 应用入口
├── agent/                     # Agent 核心模块 (LangGraph)
├── api/v1/                    # API 路由层
├── models/                    # 数据模型层 (SQLModel)
├── repositories/              # 数据访问层
├── services/                  # 业务逻辑层
├── schemas/                   # Pydantic 模式
├── llm/                       # LLM 服务
├── vector_stores/             # 向量存储
├── retrievers/                # 检索器
├── infra/                     # 基础设施
├── tasks/                     # 异步任务 (Celery)
├── middleware/                # HTTP 中间件
├── observability/             # 可观测性
├── rate_limit/                # 请求限流
├── auth/                      # 认证授权
├── config/                    # 配置管理
├── evaluation/                # 评估模块
└── utils/                     # 工具函数
```

---

## 核心模块详解

### 1. Agent 模块 (`app/agent/`)

Agent 模块是 Kiki 的核心，基于 LangGraph 实现工作流编排。

| 组件 | 文件 | 职责 |
|------|------|------|
| **主类** | `agent.py` | `LangGraphAgent` 主类，对话管理 |
| **工厂** | `factory.py` | Agent 实例工厂 |
| **状态** | `state.py` | 状态定义和管理 |
| **工作流** | `workflow.py` | 工作流定义 |
| **上下文** | `context.py` | 上下文管理 |

#### 子目录

| 目录 | 职责 |
|------|------|
| `graph/` | LangGraph 图构建 |
| `callbacks/` | 回调处理 |
| `memory/` | 记忆系统 (短期/长期/上下文) |
| `streaming/` | 流式响应处理 |
| `tools/` | 工具注册和执行 |
| `prompts/` | 提示词模板 |
| `retry/` | 重试机制 |

### 2. API 路由层 (`app/api/v1/`)

REST API 端点定义，处理 HTTP 请求。

| 端点文件 | 职责 |
|----------|------|
| `chat.py` | 聊天接口 (同步/流式) |
| `sessions.py` | 会话管理 |
| `tasks.py` | 任务管理 |
| `knowledge.py` | 知识库接口 |
| `auth.py` | 认证接口 |
| `agents.py` | Agent 配置 |
| `mcp_services.py` | MCP 服务管理 |
| `chunks.py` | 文档块管理 |
| `faq.py` | FAQ 管理 |

### 3. 数据模型层 (`app/models/`)

SQLModel 数据库模型定义。

| 模型文件 | 职责 |
|----------|------|
| `database.py` | 数据库基础模型 |
| `knowledge.py` | 知识库相关模型 |
| `task.py` | 任务模型 |
| `message.py` | 消息模型 |
| `session.py` | 会话模型 |
| `thread.py` | 线程模型 |
| `user.py` | 用户模型 |

### 4. 数据访问层 (`app/repositories/`)

Repository 模式封装数据库操作。

| Repository | 职责 |
|------------|------|
| `base.py` | Repository 基类 |
| `knowledge.py` | 知识库访问 |
| `message.py` | 消息访问 |
| `chunk.py` | 文档块访问 |
| `session.py` | 会话访问 |
| `user.py` | 用户访问 |
| `api_key.py` | API Key 访问 |
| `mcp_service.py` | MCP 服务访问 |

### 5. 业务服务层 (`app/services/`)

核心业务逻辑实现。

| 服务目录 | 职责 |
|----------|------|
| `agent/` | Agent 业务逻辑 |
| `chat/` | 聊天服务 |
| `llm/` | LLM 调用服务 |
| `knowledge/` | 知识库服务 |
| `search/` | 搜索服务 |
| `core/` | 核心服务 |
| `shared/` | 共享服务 |
| `web/` | Web 服务 |

### 6. LLM 模块 (`app/llm/`)

大语言模型服务层。

| 组件 | 文件 | 职责 |
|------|------|------|
| **服务** | `service.py` | LLM 调用接口 |
| **注册表** | `registry.py` | 模型注册表 |
| **提供商** | `providers.py` | 模型提供商适配 |
| **成本追踪** | `cost_tracker.py` | Token 成本统计 |
| **Embeddings** | `embeddings.py` | 向量化服务 |

### 7. 向量存储 (`app/vector_stores/`)

向量数据库抽象层，支持多种后端。

| 实现 | 职责 |
|------|------|
| `base.py` | 存储基类 |
| `factory.py` | 存储工厂 |
| `qdrant.py` | Qdrant 实现 |
| `pgvector.py` | PGVector 实现 |
| `pinecone.py` | Pinecone 实现 |
| `elasticsearch.py` | Elasticsearch 实现 |

### 8. 检索器 (`app/retrievers/`)

RAG 检索器实现。

| 检索器 | 职责 |
|--------|------|
| `base.py` | 检索器基类 |
| `elasticsearch.py` | ES 检索 |
| `conversational.py` | 对话式检索 |
| `ensemble.py` | 集成检索 |
| `bm25.py` | BM25 关键词检索 |

### 9. 基础设施 (`app/infra/`)

底层基础设施服务。

| 组件 | 文件 | 职责 |
|------|------|------|
| **数据库** | `database.py` | 数据库连接池 |
| **Redis** | `redis.py` | Redis 客户端 |
| **缓存** | `cache.py` | 缓存服务 |
| **存储** | `storage.py` | 文件存储 |
| **搜索** | `search.py` | 搜索基础设施 |

### 10. 异步任务 (`app/tasks/`)

Celery 异步任务系统。

| 组件 | 文件 | 职责 |
|------|------|------|
| **应用** | `celery_app.py` | Celery 配置 |
| **存储** | `store.py` | 任务存储 |
| **处理器** | `handlers/` | 任务处理器 |

#### 任务处理器

| 处理器 | 职责 |
|--------|------|
| `document.py` | 文档处理 |
| `chunk.py` | 分块处理 |
| `generation.py` | 生成任务 |
| `faq.py` | FAQ 生成 |
| `cleanup.py` | 清理任务 |
| `kb_clone.py` | 知识库克隆 |

### 11. 中间件 (`app/middleware/`)

HTTP 中间件。

| 中间件 | 职责 |
|--------|------|
| `auth.py` | 认证中间件 |
| `security.py` | 安全中间件 |
| `observability.py` | 可观测性中间件 |

### 12. 可观测性 (`app/observability/`)

日志、指标、审计。

| 组件 | 职责 |
|------|------|
| `logging.py` | 结构化日志 |
| `metrics.py` | Prometheus 指标 |
| `audit.py` | 审计日志 |
| `elk_handler.py` | ELK 日志处理 |
| `log_sanitizer.py` | 日志脱敏 |

### 13. 认证授权 (`app/auth/`)

认证模块。

| 组件 | 文件 | 职责 |
|------|------|------|
| **JWT** | `jwt.py` | JWT Token 处理 |
| **API Key** | `api_key.py` | API Key 认证 |
| **租户** | `tenant.py` | 租户认证 |
| **租户 Key** | `tenant_api_key.py` | 租户 API Key |

### 14. 配置管理 (`app/config/`)

| 组件 | 文件 | 职责 |
|------|------|------|
| **设置** | `settings.py` | Pydantic Settings |
| **运行时** | `runtime.py` | 运行时配置 |
| **依赖** | `dependencies.py` | 依赖注入 |

---

## 数据流

```
Client Request
    ↓
API Layer (app/api/v1/)
    ↓
Middleware (auth, security, observability)
    ↓
Service Layer (app/services/)
    ↓
Repository Layer (app/repositories/)
    ↓
Database (PostgreSQL + Redis)
```

```
Agent Execution Flow
    ↓
Agent (app/agent/)
    ↓
Graph Builder (graph/builder.py)
    ↓
Nodes (nodes.py) → Tools (tools/)
    ↓
LLM Service (llm/service.py)
    ↓
Vector Stores (vector_stores/)
    ↓
Retrievers (retrievers/)
```

---

## 关键入口点

### 应用入口

```python
# app/main.py
from app.main import create_app

app = create_app()
```

### Agent 入口

```python
# app/agent/agent.py
from app.agent import LangGraphAgent

agent = LangGraphAgent()
```

### 配置加载

```python
# app/config/settings.py
from app.config import settings

print(settings.database_url)
```

---

## 模块依赖关系

```
main.py
├── api/          ← agent/, services/, schemas/
├── agent/        ← llm/, vector_stores/, tools/
├── services/     ← repositories/, llm/, search/
├── repositories/ ← models/, infra/
├── models/       ← SQLModel
├── schemas/      ← pydantic
├── llm/          ← langchain/openai
├── infra/        ← redis, asyncpg
├── middleware/   ← auth/, observability/
├── auth/         ← jwt
├── observability/← structlog, prometheus
├── config/       ← pydantic-settings
└── tasks/        ← celery
```

---

## 相关文档

- [README](../README.md) - 项目概览
- [API](API.md) - API 接口文档
- [AGENT](AGENT.md) - Agent 使用指南
- [ARCHITECTURE](ARCHITECTURE.md) - 系统架构
