# 企业级 Agent 功能使用指南

## 概述

本文档介绍 Kiki Agent 框架中所有企业级功能的使用方法。

---

## 目录

1. [限流器](#1-限流器)
2. [指标监控](#2-指标监控)
3. [认证系统](#3-认证系统)
4. [多 Agent 系统](#4-多-agent-系统)
5. [Web 搜索](#5-web-搜索)
6. [LangSmith 集成](#6-langsmith-集成)
7. [MCP 工具集成](#7-mcp-工具集成)

---

## 1. 限流器

### 基础使用

```python
from fastapi import APIRouter
from app.core.limiter import limiter, RateLimit

router = APIRouter()

# 使用预定义限流
@router.get("/chat")
@limiter.limit(RateLimit.CHAT)  # 30 per minute, 500 per day
async def chat(message: str):
    return {"response": "..."}

# 自定义限流
@router.post("/api/action")
@limiter.limit("10 per hour")
async def special_action():
    pass
```

### 预定义限流规则

| 规则 | 限制 |
|------|------|
| `RateLimit.CHAT` | 30/分钟, 500/天 |
| `RateLimit.CHAT_STREAM` | 20/分钟, 300/天 |
| `RateLimit.REGISTER` | 10/小时, 50/天 |
| `RateLimit.LOGIN` | 20/分钟, 100/天 |
| `RateLimit.API` | 100/分钟, 1000/天 |

### 按 用户限流

```python
# 限流器自动识别用户（从请求头 X-User-ID 或 JWT）
@router.get("/user/profile")
@limiter.limit("60 per minute")
async def get_profile(user_id: str):
    # 每个用户独立限流
    pass
```

---

## 2. 指标监控

### HTTP 指标（自动收集）

```python
from app.core.metrics import setup_metrics

# 在 main.py 中添加
setup_metrics(app)

# 访问 /metrics 查看 Prometheus 格式指标
```

### 自定义指标

```python
from app.core.metrics import (
    track_llm_request,
    record_llm_tokens,
    increment_active_sessions,
)

# 追踪 LLM 调用
async with track_llm_request(model="gpt-4o", provider="openai"):
    response = await llm.ainvoke(messages)

# 记录 Token 使用
record_llm_tokens("gpt-4o", prompt_tokens=100, completion_tokens=50)

# 更新活跃会话数
increment_active_sessions(1)   # 增加
increment_active_sessions(-1)  # 减少
```

### 可用指标

| 指标名 | 类型 | 标签 |
|--------|------|------|
| `http_requests_total` | Counter | method, endpoint, status |
| `http_request_duration_seconds` | Histogram | method, endpoint |
| `llm_requests_total` | Counter | model, provider, status |
| `llm_duration_seconds` | Histogram | model, provider |
| `llm_tokens_total` | Counter | model, token_type |
| `tool_calls_total` | Counter | tool_name, status |
| `active_sessions` | Gauge | - |
| `active_users` | Gauge | - |

---

## 3. 认证系统

### 创建 Token

```python
from app.core.auth import create_access_token, Token

# 创建访问令牌
token: Token = create_access_token(
    data={"sub": "user-123", "role": "user"},
    expires_delta=timedelta(hours=1),
)

print(token.access_token)
print(token.expires_at)
```

### 验证 Token

```python
from app.core.auth import verify_token, get_token_sub

# 验证并获取用户 ID
user_id = get_token_sub(token.access_token)
if user_id:
    print(f"用户: {user_id}")
```

### FastAPI 依赖注入

```python
from fastapi import Depends
from app.core.auth import get_current_user_id

@router.get("/protected")
async def protected_route(
    user_id: str = Depends(get_current_user_id),
):
    return {"user_id": user_id}
```

### 数据库用户模型

```python
from app.models.database import User, UserCreate

# 创建新用户
user = User(
    email="user@example.com",
    full_name="John Doe",
)
user.set_password("secure_password")

# 验证密码
if user.verify_password("input_password"):
    print("密码正确")
```

---

## 4. 多 Agent 系统

### Router Agent（路由模式）

```python
from app.agent.multi_agent import RouterAgent, create_multi_agent_system
from app.agent.graph import compile_chat_graph

# 创建专业 Agent
sales_agent = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是销售专家...",
)

support_agent = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是客服专家...",
)

# 创建路由系统
router = create_multi_agent_system(
    mode="router",
    llm_service=llm_service,
    agents={
        "Sales": sales_agent,
        "Support": support_agent,
    },
)

# 使用
response = await router.ainvoke(
    {"messages": [{"role": "user", "content": "我想买手机"}]},
    config={"configurable": {"thread_id": "session-1"}},
)
```

### Supervisor Agent（监督模式）

```python
from app.agent.multi_agent import SupervisorAgent

# 创建 Worker Agent
researcher = compile_chat_graph(llm_service, system_prompt="研究员...")
writer = compile_chat_graph(llm_service, system_prompt="写手...")
reviewer = compile_chat_graph(llm_service, system_prompt="审核员...")

# 创建监督系统
supervisor = SupervisorAgent(
    llm_service=llm_service,
    workers={
        "Researcher": researcher,
        "Writer": writer,
        "Reviewer": reviewer,
    },
)

graph = supervisor.graph.compile()
```

### Handoff Agent（Swarm 模式）

```python
from app.agent.multi_agent import HandoffAgent, create_swarm

# 创建可切换 Agent
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

# 创建 Swarm
swarm = create_swarm(
    agents=[alice, bob],
    default_agent="Alice",
)
```

---

## 5. Web 搜索

### 使用 DuckDuckGo 搜索

```python
from app.core.agent.tools.search import search_web

# 直接调用
results = await search_web.func(
    query="Python LangGraph 教程",
    max_results=5,
)
print(results)
```

### 统一搜索接口

```python
from app.core.agent.tools.search import get_search_engine

engine = get_search_engine()

# 自动尝试多个搜索引擎（自动回退）
results = await engine.search(
    query="最新 AI 新闻",
    max_results=5,
)
```

### 为 Agent 添加搜索能力

```python
from app.core.agent.tools.search import with_web_search

@with_web_search
class SearchEnabledAgent:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    async def process(self, query: str):
        # 现在可以调用 self.search_web()
        search_results = await self.search_web(query)
        ...
```

---

## 6. LangSmith 集成

### 自动追踪

```python
from app.core.observability import get_langsmith_callbacks, get_run_config

# 添加回调
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

### 数据集管理

```python
from app.core.observability import DatasetManager

# 创建数据集
dataset = DatasetManager("chatbot_eval")

# 添加测试用例
dataset.add_example(
    input_text="你好",
    expected_output="你好！有什么我可以帮助你的吗？",
)

# 批量添加
examples = [
    {"input": "2+2等于几？", "output": "4"},
    {"input": "讲个笑话", "output": None},
]
dataset.add_examples_from_list(examples)
```

### 运行追踪

```python
from app.core.observability import LangSmithClient

with LangSmithClient.trace_run(project_name="MyProject"):
    result = await agent.ainvoke(input_data)
```

---

## 7. MCP 工具集成

### 注册 MCP 服务器

```python
from app.core.mcp import MCPRegistry, load_mcp_tools

# 注册 uvx 启动的服务器
MCPRegistry.register(
    name="filesystem",
    command="uvx",
    args=["mcp-server-filesystem", "/allowed/path"],
)

# 注册 npx 启动的服务器
MCPRegistry.register(
    name="github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
)

# 注册 HTTP 服务器
MCPRegistry.register(
    name="custom",
    transport="http",
    url="http://localhost:3000/mcp",
)
```

### 加载 MCP 工具

```python
# 初始化所有服务器并加载工具
mcp_tools = await load_mcp_tools()

# 绑定到 Agent
agent.bind_tools(mcp_tools)
```

### 预定义的 MCP 服务器

```python
from app.core.mcp import list_predefined_mcp_servers, get_predefined_mcp_server

# 查看可用服务器
servers = list_predefined_mcp_servers()
print(servers)  # ['filesystem', 'github', 'postgres', 'brave-search', 'fetch']

# 使用预定义配置
config = get_predefined_mcp_server("github")
MCPRegistry.register(**config)
```

### 动态调用 MCP 工具

```python
from app.core.mcp import MCPRegistry

client = MCPRegistry.get("filesystem")

if client:
    # 初始化连接
    await client.initialize()

    # 调用工具
    result = await client.call_tool(
        tool_name="read_file",
        arguments={"path": "/allowed/path/file.txt"},
    )

    # 获取所有 LangChain 格式的工具
    tools = client.get_langchain_tools()
```

---

## 快速开始

### 完整示例：带限流、认证和追踪的聊天 API

```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.limiter import limiter, RateLimit
from app.core.auth import create_access_token, get_current_user_id
from app.core.observability import get_langsmith_callbacks, get_run_config
from app.core.agent import get_agent

router = APIRouter(prefix="/chat", tags=["chat"])

# 登录获取 Token
@router.post("/login")
@limiter.limit(RateLimit.LOGIN)
async def login(email: str, password: str):
    # 验证用户（省略）
    if not verify_user(email, password):
        raise HTTPException(status_code=401, detail="认证失败")

    token = create_access_token(data={"sub": email})
    return {"access_token": token.access_token}

# 聊天接口（带认证、限流、追踪）
@router.post("")
@limiter.limit(RateLimit.CHAT)
async def chat(
    message: str,
    user_id: str = Depends(get_current_user_id),
):
    agent = await get_agent()

    callbacks = get_langsmith_callbacks()

    response = await agent.get_response(
        message=message,
        session_id=f"{user_id}-default",
    )

    return {"response": response[-1].content}

# 流式聊天
@router.post("/stream")
@limiter.limit(RateLimit.CHAT_STREAM)
async def chat_stream(
    message: str,
    user_id: str = Depends(get_current_user_id),
):
    agent = await get_agent()

    async def generate():
        async for chunk in agent.get_stream_response(
            message=message,
            session_id=f"{user_id}-default",
        ):
            yield f"data: {json.dumps({'content': chunk})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 环境变量配置

```bash
# LangSmith 追踪
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="lsv2_..."
export LANGCHAIN_PROJECT="kiki-agent"

# OpenAI（或其他 LLM）
export OPENAI_API_KEY="sk-..."
export KIKI_LLM_MODEL="gpt-4o"

# 数据库
export KIKI_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/kiki"

# JWT
export KIKI_SECRET_KEY="your-secret-key-min-32-characters"
```

---

## 更多信息

- [多 Agent 架构指南](./MULTI_AGENT_GUIDE.md)
- [LangSmith 文档](https://docs.smith.langchain.com/)
- [MCP 规范](https://modelcontextprotocol.io/)
