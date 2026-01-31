# Kiki Agent Framework

> 企业级 Python Agent 开发脚手架 - 基于 FastAPI + LangGraph

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-orange.svg)](https://langchain-ai.github.io/langgraph/)

## 项目简介

Kiki Agent Framework 是一个生产就绪的 AI Agent 开发框架，旨在简化企业级 AI 应用的构建。它集成了最新的 LangGraph 工作流编排能力，提供了完整的认证授权、监控、日志、限流等企业级功能。

### 核心特性

#### AI 原生设计
- **LangGraph 深度集成** - 状态管理、检查点持久化、流式响应
- **多 Agent 协作** - Router、Supervisor、Swarm 模式
- **工具调用系统** - 内置工具注册表，支持 MCP 协议
- **多模型支持** - OpenAI、Anthropic、DeepSeek、DashScope、Ollama

#### 企业级功能
- **认证授权** - JWT + RBAC 权限管理
- **请求限流** - 基于 Redis 的令牌桶算法
- **结构化日志** - structlog + ContextVar 上下文绑定
- **可观测性** - Langfuse 追踪 + Prometheus 指标
- **错误处理** - 统一错误分类和用户友好提示

#### 开发体验
- **类型安全** - 完整的 mypy 类型检查
- **热重载** - 开发环境自动重启
- **测试支持** - pytest + pytest-asyncio，80%+ 覆盖率目标
- **容器化** - Docker + Docker Compose 一键部署

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 高性能异步 Web 框架 |
| Agent 框架 | LangGraph + LangChain | 工作流编排和状态管理 |
| 数据库 | PostgreSQL + asyncpg | 关系型数据库 + pgvector |
| 缓存 | Redis | 分布式缓存和会话存储 |
| 日志 | structlog | 结构化日志 |
| 追踪 | Langfuse | LLM 调用追踪和评估 |
| 指标 | Prometheus + Grafana | 监控指标收集和可视化 |
| 包管理 | uv | 快速 Python 包管理器 |

## 快速开始

### 环境要求

- Python 3.13+
- PostgreSQL 14+
- Redis 6+
- Docker (可选)

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-org/kiki.git
cd kiki
```

2. **安装依赖**
```bash
uv sync
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，设置必要的配置
```

4. **启动服务**
```bash
# 开发环境
uv run uvicorn app.main:app --reload

# 或使用 Docker
make up
```

5. **访问服务**
- API 文档: http://localhost:8000/docs
- Grafana 监控: http://localhost:3000
- Prometheus 指标: http://localhost:9090

## 项目结构

```
kiki/
├── app/                      # 应用代码
│   ├── api/                  # API 路由层
│   │   └── v1/              # API v1 版本
│   ├── agent/               # Agent 核心模块
│   │   ├── graphs/          # LangGraph 工作流
│   │   ├── tools/           # 工具注册表
│   │   └── memory/          # 记忆管理
│   ├── core/                # 核心模块
│   │   ├── config.py        # 配置管理
│   │   ├── auth.py          # 认证授权
│   │   ├── middleware.py    # 中间件
│   │   └── errors.py        # 错误处理
│   ├── llm/                 # LLM 服务
│   │   ├── service.py       # LLM 服务实现
│   │   ├── providers.py     # 模型提供商
│   │   └── registry.py      # 模型注册表
│   ├── infra/               # 基础设施
│   │   ├── redis.py         # Redis 客户端
│   │   └── storage.py       # 对象存储
│   ├── observability/       # 可观测性
│   │   ├── logging.py       # 日志配置
│   │   └── metrics.py       # 指标收集
│   ├── models/              # 数据模型
│   ├── schemas/             # Pydantic 模式
│   ├── repositories/        # 数据访问层
│   └── main.py              # 应用入口
├── tests/                   # 测试目录
├── docs/                    # 文档目录
├── config/                  # 配置文件
├── scripts/                 # 脚本目录
└── pyproject.toml          # 项目配置
```

## 快速示例

### 基础聊天

```python
from app.agent import get_agent

# 获取 Agent 实例
agent = await get_agent()

# 发送消息
response = await agent.get_response(
    message="你好，请介绍一下自己",
    session_id="user-123",
)

print(response[-1].content)
```

### 流式响应

```python
from app.agent import get_agent

agent = await get_agent()

async for chunk in agent.get_stream_response(
    message="写一首关于春天的诗",
    session_id="user-123",
):
    print(chunk, end="", flush=True)
```

### 自定义工具

```python
from app.agent.tools import tool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(..., description="城市名称")

@tool
async def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 实现天气查询逻辑
    return f"{city} 今天晴朗，温度 25°C"

# 工具自动注册到全局注册表
```

### 创建 Router Agent

```python
from app.agent.multi_agent import create_router_agent

router_agent = create_router_agent(
    agents={
        "sales": sales_agent,
        "support": support_agent,
        "general": general_agent,
    }
)

response = await router_agent.ainvoke({
    "messages": [("user", "我需要退款帮助")],
})
```

## API 文档

### 聊天接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/chat` | POST | 发送消息（同步） |
| `/api/v1/chat/stream` | POST | 发送消息（流式 SSE） |
| `/api/v1/chat/history/{session_id}` | GET | 获取聊天历史 |
| `/api/v1/chat/history/{session_id}` | DELETE | 清空聊天历史 |

### 认证接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/auth/register` | POST | 用户注册 |
| `/api/v1/auth/login` | POST | 用户登录 |
| `/api/v1/auth/refresh` | POST | 刷新 Token |

### Agent 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/agents/router` | POST | 创建路由 Agent |
| `/api/v1/agents/supervisor` | POST | 创建监督 Agent |
| `/api/v1/agents/swarm` | POST | 创建 Swarm Agent |

## 配置说明

### 环境变量

```bash
# 应用配置
KIKI_ENV=development                    # 环境: development/staging/production/test
KIKI_SECRET_KEY=your-secret-key         # JWT 密钥（生产环境至少 32 字符）

# 数据库
KIKI_DATABASE_URL=postgresql+asyncpg://localhost:5432/kiki

# Redis
KIKI_REDIS_URL=redis://localhost:6379/0

# LLM 配置
KIKI_LLM_PROVIDER=openai                # 提供商: openai/anthropic/deepseek/dashscope/ollama
KIKI_LLM_MODEL=gpt-4o                   # 模型名称
KIKI_LLM_API_KEY=your-api-key

# Langfuse（可选）
KIKI_LANGCHAIN_TRACING_V2=true
KIKI_LANGCHAIN_API_KEY=your-langfuse-key
```

## 部署指南

详细的部署指南请参考 [DEPLOYMENT.md](../DEPLOYMENT.md)。

### Docker 部署

```bash
# 启动所有服务
make up-all

# 查看日志
make logs

# 停止服务
make down
```

## 开发指南

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行测试并生成覆盖率报告
uv run pytest --cov=app --cov-report=html

# 运行特定测试
uv run pytest tests/unit/test_config.py
```

### 代码检查

```bash
# 代码格式检查
uv run ruff check .

# 自动修复
uv run ruff check --fix .

# 类型检查
uv run mypy app/
```

## 文档

- [架构设计](ARCHITECTURE.md) - 系统架构和设计模式
- [API 文档](API.md) - 完整的 API 接口文档
- [Agent 系统](AGENT.md) - Agent 使用指南
- [开发指南](DEVELOPMENT.md) - 开发规范和贡献指南
- [部署指南](../DEPLOYMENT.md) - 生产环境部署
- [路线图](ROADMAP.md) - 版本规划和后续功能
- [可观测性增强](OBSERVABILITY_GUIDE.md) - ELK、Prometheus、限流、缓存

## 许可证

MIT License

## 贡献

欢迎贡献！请阅读 [DEVELOPMENT.md](DEVELOPMENT.md) 了解如何参与贡献。
