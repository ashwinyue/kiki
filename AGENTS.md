# 企业级 Agent 开发脚手架指南

> 基于 FastAPI + LangGraph 的企业级 Agent 开发脚手架，遵循 LangChain/LangGraph 最佳实践。

---

## 项目概述

Kiki 是一个企业级 Agent 开发脚手架，提供：
- **LangGraph 原生集成** - StateGraph + Command/Conditional Edges + Checkpoint 持久化
- **多 Agent 协作模式** - Router、Supervisor、Handoff (Swarm)
- **企业级特性** - 限流、认证、指标监控、可观测性
- **工具生态** - MCP 协议集成、Web 搜索、自定义工具

---

## 技术栈

### 核心依赖
```toml
# Web Framework
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.10.0
pydantic-settings>=2.6.0

# LangGraph & LangChain
langgraph>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0

# Database
sqlmodel>=0.0.24
asyncpg>=0.30.0
langgraph-checkpoint-postgres>=2.0.0

# Observability
structlog>=25.1.0
prometheus-client>=0.21.0

# Auth
python-jose[cryptography]>=3.4.0
passlib[bcrypt]>=1.7.4
slowapi>=0.1.9

# Utils
python-dotenv>=1.0.0
tenacity>=9.1.0
```

---

## 项目目录结构

```
kiki/
├── app/                              # 主应用代码
│   ├── api/                          # API 路由层
│   │   ├── __init__.py
│   │   └── deps.py                   # FastAPI 依赖注入
│   ├── core/                         # 核心模块
│   │   ├── agent/                    # Agent 核心模块
│   │   │   ├── __init__.py
│   │   │   ├── state.py              # Agent 状态 (add_messages reducer)
│   │   │   ├── graph.py              # LangGraph 工作流
│   │   │   ├── tools.py              # 工具定义 (@tool 装饰器)
│   │   │   ├── multi_agent.py        # 多 Agent 协作
│   │   │   └── MULTI_AGENT_GUIDE.md  # 多 Agent 使用指南
│   │   ├── llm.py                    # LLM 服务 (多提供商、重试、结构化输出)
│   │   ├── config.py                 # 配置管理
│   │   ├── logging.py                # 结构化日志 (structlog)
│   │   ├── limiter.py                # 速率限制 (slowapi)
│   │   ├── metrics.py                # Prometheus 指标
│   │   ├── auth.py                   # JWT 认证
│   │   ├── search.py                 # Web 搜索工具
│   │   ├── mcp.py                    # MCP 协议集成
│   │   └── observability.py          # LangSmith 集成
│   ├── models/                       # 数据模型 (SQLModel)
│   │   ├── __init__.py
│   │   └── database.py               # 用户、会话、线程、消息模型
│   └── main.py                       # 应用入口
├── tests/                            # 测试套件
├── .env.example                      # 环境变量示例
├── AGENTS.md                         # 本文档
├── ENTERPRISE_GUIDE.md               # 企业级功能使用指南
└── pyproject.toml                    # 项目配置
```

---

## 核心架构

### 1. Agent 状态管理 (LangGraph 最佳实践)

```python
# app/core/agent/state.py
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Agent 状态定义

    使用 add_messages reducer 自动管理消息历史，
    LangGraph 会自动处理消息的追加而不是覆盖。
    """
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str | None
    session_id: str | None
```

### 2. LLM 服务 (重试 + 结构化输出)

```python
# app/core/llm.py
from app.core.llm import get_llm_service, LLMService

llm_service = get_llm_service()

# 普通调用 (内置重试和回退)
response = await llm_service.ainvoke(messages)

# 结构化输出 (用于路由决策等场景)
from pydantic import BaseModel

class RouteDecision(BaseModel):
    agent: str = Field(description="目标 agent 名称")
    reason: str = Field(description="选择原因")
    confidence: float = Field(ge=0.0, le=1.0)

structured_llm = llm_service.with_structured_output(RouteDecision)
decision: RouteDecision = await structured_llm.ainvoke(messages)
```

### 3. Agent Graph (条件路由)

```python
# app/core/agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langchain_core.prompts import ChatPromptTemplate

def route_by_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """条件路由函数

    根据最后一条消息是否有 tool_calls 决定下一步。
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"

# 构建图
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("agent", route_by_tools)
builder.set_entry_point("agent")
graph = builder.compile(checkpointer=checkpointer)
```

### 4. 工具定义 (@tool 装饰器)

```python
# app/core/agent/tools.py
from langchain_core.tools import tool

@tool
async def search_web(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 搜索网络

    Args:
        query: 搜索查询
        max_results: 最大结果数
    """
    # 实现逻辑
    return results

# 绑定到 LLM
llm_with_tools = llm.bind_tools([search_web])
```

---

## 多 Agent 协作模式

### Router Agent (路由模式)

适用于意图分类、任务分发场景。

```python
from app.core.agent.multi_agent import RouterAgent, create_multi_agent_system
from app.core.agent.graph import AgentGraph

# 创建专业 Agent
sales_agent = AgentGraph(llm_service, system_prompt="销售专家...")
support_agent = AgentGraph(llm_service, system_prompt="客服专家...")

# 创建路由系统
router_system = create_multi_agent_system(
    mode="router",
    llm_service=llm_service,
    agents={
        "Sales": sales_agent,
        "Support": support_agent,
    },
)
```

### Supervisor Agent (监督模式)

适用于复杂任务分解、多步骤协作。

```python
from app.core.agent.multi_agent import SupervisorAgent

supervisor = SupervisorAgent(
    llm_service=llm_service,
    workers={
        "Researcher": researcher,
        "Writer": writer,
        "Reviewer": reviewer,
    },
)
```

### Handoff Agent (Swarm 模式)

适用于动态协作、Agent 自主切换。

```python
from app.core.agent.multi_agent import HandoffAgent, create_swarm

alice = HandoffAgent(
    name="Alice",
    llm_service=llm_service,
    tools=[search_products],
    handoff_targets=["Bob"],
)

bob = HandoffAgent(
    name="Bob",
    llm_service=llm_service,
    tools=[check_specifications],
    handoff_targets=["Alice"],
)

swarm = create_swarm(agents=[alice, bob], default_agent="Alice")
```

> 详见 [MULTI_AGENT_GUIDE.md](./app/core/agent/MULTI_AGENT_GUIDE.md)

---

## 企业级功能

### 1. 速率限制

```python
from app.core.limiter import limiter, RateLimit

@router.post("/chat")
@limiter.limit(RateLimit.CHAT)  # 30/min, 500/day
async def chat(message: str):
    return {"response": "..."}
```

### 2. JWT 认证

```python
from app.core.auth import create_access_token, get_current_user_id

token = create_access_token(data={"sub": "user-123"})

@router.get("/protected")
async def protected(user_id: str = Depends(get_current_user_id)):
    return {"user_id": user_id}
```

### 3. Prometheus 指标

```python
from app.core.metrics import track_llm_request, record_llm_tokens

async with track_llm_request(model="gpt-4o", provider="openai"):
    response = await llm.ainvoke(messages)

record_llm_tokens("gpt-4o", prompt_tokens=100, completion_tokens=50)
```

### 4. LangSmith 可观测性

```python
from app.core.observability import get_langsmith_callbacks, get_run_config

callbacks = get_langsmith_callbacks()

response = await graph.ainvoke(
    input_data,
    config=get_run_config(
        run_name="chat_session_123",
        metadata={"user_id": "user-123"},
    ),
    config={"callbacks": callbacks},
)
```

### 5. MCP 工具集成

```python
from app.core.mcp import MCPRegistry, load_mcp_tools

# 注册 MCP 服务器
MCPRegistry.register(
    name="filesystem",
    command="uvx",
    args=["mcp-server-filesystem", "/allowed/path"],
)

# 加载所有 MCP 工具
mcp_tools = await load_mcp_tools()
agent.bind_tools(mcp_tools)
```

### 6. Web 搜索

```python
from app.core.search import get_search_engine

engine = get_search_engine()
results = await engine.search(query="最新 AI 新闻", max_results=5)
```

> 详见 [ENTERPRISE_GUIDE.md](./ENTERPRISE_GUIDE.md)

---

## 编码规范

### 日志 (structlog)

```python
from app.core.logging import get_logger

log = get_logger(__name__)

# 正确 - 使用键值对
log.info("chat_request_received", session_id=session.id, message_count=len(messages))

# 错误 - 禁止 f-string
log.info(f"chat_request_received {session.id}")

# 异常必须用 exception() 保留堆栈
log.exception("llm_call_failed", error=str(e))
```

### 重试 (tenacity)

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def call_llm(messages: list) -> str:
    # LLM 调用逻辑
    ...
```

### 错误处理

```python
async def process_request(request: Request) -> Response:
    # 1. 前置验证（卫语句）
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # 2. 快乐路径
    try:
        result = await agent.process(request.messages)
        log.info("request_processed_successfully")
        return result
    except SpecificError as e:
        log.error("specific_error_occurred", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("unexpected_error")
        raise HTTPException(status_code=500, detail="Internal error")
```

---

## 数据库模型

```python
# app/models/database.py
from app.models.database import User, Session, Thread, Message

# 用户
user = User(
    email="user@example.com",
    full_name="John Doe",
)
user.set_password("secure_password")

# 会话 (避免与 SQLAlchemy.Session 冲突，命名为 ChatSession)
session = Session(id=str(uuid.uuid4()), user_id=user.id, name="新对话")

# 消息
message = Message(
    session_id=session.id,
    role="user",
    content="你好",
)
```

---

## 环境变量配置

```bash
# 应用配置
KIKI_APP_NAME=Kiki Agent
KIKI_ENVIRONMENT=development
KIKI_DEBUG=true

# 数据库配置
KIKI_DATABASE_URL=postgresql+asyncpg://localhost:5432/kiki

# LLM 配置
KIKI_LLM__PROVIDER=openai
KIKI_LLM__MODEL=gpt-4o
KIKI_LLM__API_KEY=your-api-key

# 认证配置
KIKI_SECRET_KEY=change-me-in-production-min-32-chars
KIKI_ACCESS_TOKEN_EXPIRE_MINUTES=30

# 可观测性配置
KIKI_LOG_LEVEL=INFO

# LangSmith (替代 Langfuse)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=kiki-agent
```

---

## 十大工程原则

1. **所有路由必须配置速率限制**
2. **所有 LLM 调用必须启用可观测性追踪**
3. **所有异步操作必须有完整错误处理**
4. **所有日志必须遵循结构化格式 (lowercase_underscore)**
5. **所有重试必须使用 tenacity 库**
6. **所有敏感配置必须通过环境变量管理**
7. **所有服务必须通过依赖注入解耦**
8. **所有检查点必须持久化到 PostgreSQL**
9. **Agent 状态必须使用 add_messages reducer**
10. **工具定义必须使用 @tool 装饰器**

---

## 参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangChain 官方文档](https://python.langchain.com/docs/)
- [FastAPI 最佳实践](https://fastapi.tiangolo.com/tutorial/)
- [MCP 协议规范](https://modelcontextprotocol.io/)
- [LangSmith 文档](https://docs.smith.langchain.com/)
