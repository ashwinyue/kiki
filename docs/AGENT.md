# Kiki Agent Framework - Agent 系统文档

> 版本: v0.1.0
> 更新日期: 2025-01-30

## 目录

- [概述](#概述)
- [Agent 基础](#agent-基础)
- [工作流图类型](#工作流图类型)
- [工具系统](#工具系统)
- [多 Agent 协作](#multi-agent-协作)
- [状态管理](#状态管理)
- [检查点持久化](#检查点持久化)
- [最佳实践](#最佳实践)

---

## 概述

Kiki Agent 系统基于 LangGraph 构建，提供了完整的 AI Agent 开发能力：

- **单 Agent 对话** - 基础对话，支持工具调用
- **多 Agent 协作** - Router、Supervisor、Swarm 模式
- **状态管理** - 持久化会话状态
- **工具调用** - 灵活的工具注册和执行
- **流式响应** - 实时流式输出

### 核心概念

```
┌─────────────────────────────────────────────────────┐
│                    LangGraph                         │
│  ┌───────────────────────────────────────────────┐  │
│  │                 StateGraph                     │  │
│  │  ┌─────────┐      ┌─────────┐      ┌─────────┐│  │
│  │  │  Node   │ ──▶  │  Node   │ ──▶  │  Node   ││  │
│  │  └─────────┘      └─────────┘      └─────────┘│  │
│  │       ▲                │                │      │  │
│  │       └────────────────┴────────────────┘      │  │
│  │                   Conditional Edge              │  │
│  └───────────────────────────────────────────────┘  │
│                      │                                │
│                      ▼                                │
│  ┌───────────────────────────────────────────────┐  │
│  │              Checkpointer                      │  │
│  │  - 持久化状态                                   │  │
│  │  - 时间旅行                                     │  │
│  │  - 恢复执行                                     │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Agent 基础

### 创建 Agent

```python
from app.agent import LangGraphAgent, create_agent
from app.llm import get_llm_service

# 获取 LLM 服务
llm_service = get_llm_service()

# 方式 1: 使用 LangGraphAgent 类
agent = LangGraphAgent(
    llm_service=llm_service,
    system_prompt="你是一个专业的技术顾问"
)

# 方式 2: 直接编译 Chat Graph
from app.agent.graph import compile_chat_graph

compiled_graph = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是一个专业的技术顾问"
)
```

### 多 Agent 系统

Kiki 提供了三种多 Agent 协作模式：

```python
from app.agent.multi_agent import (
    RouterAgent,
    SupervisorAgent,
    HandoffAgent,
    create_swarm,
    create_multi_agent_system,
)
```

### 同步对话

```python
from app.agent import get_agent

agent = await get_agent()

messages = await agent.get_response(
    message="解释什么是 LangGraph",
    session_id="user-123",
)

# 获取最后一条 AI 消息
ai_message = [m for m in messages if m.type == "ai"][-1]
print(ai_message.content)
```

### 流式对话

```python
from app.agent import get_agent

agent = await get_agent()

async for chunk in agent.get_stream_response(
    message="写一首关于春天的诗",
    session_id="user-123",
):
    print(chunk, end="", flush=True)
```

### 获取聊天历史

```python
from app.agent import get_agent

agent = await get_agent()

# 获取历史消息
history = await agent.get_chat_history(session_id="user-123")

for msg in history:
    print(f"{msg.type}: {msg.content}")
```

### 清空历史

```python
from app.agent import get_agent

agent = await get_agent()
await agent.clear_chat_history(session_id="user-123")
```

---

## 工作流图类型

### Compiled Chat Graph - 通用对话图

`compile_chat_graph` 会返回编译后的 LangGraph，可用于统一的调用接口。

```python
from app.agent.graph import compile_chat_graph
from app.llm import get_llm_service

llm_service = get_llm_service()

# 编译 Agent 图
compiled = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是一个有用的助手"
)

response = await compiled.ainvoke(
    {"messages": [("user", "你好")]},
    {"configurable": {"thread_id": "session-123"}}
)
```

**支持的调用方式：**

```python
# 同步调用
result = await compiled.ainvoke(input_data, config)

# 流式调用
async for chunk in compiled.astream(input_data, config):
    print(chunk)

# 获取状态
state = await compiled.aget_state(config)
```

### 1. ChatGraph - 基础对话图

最简单的对话流程，支持工具调用。

```python
from app.agent.graph import compile_chat_graph

graph = compile_chat_graph(
    system_prompt="你是一个有用的助手"
)

response = await graph.ainvoke(
    {"messages": [("user", "你好")]},
    {"configurable": {"thread_id": "session-123"}}
)
```

**流程图：**
```
     ┌─────────┐
     │  入口   │
     └────┬────┘
          │
          ▼
    ┌──────────┐
    │  chat    │ ◀──────┐
    │  节点    │        │
    └────┬─────┘        │
         │              │
         │ 有工具调用?    │
         │              │
    ┌────┴────┐         │
    │         │         │
   YES       NO         │
    │         │         │
    ▼         ▼         │
┌─────────┐  ┌─────┐   │
│  tools  │  │ END │   │
│  节点   │  └─────┘   │
└────┬────┘            │
     │                 │
     └─────────────────┘
```

### 2. ReactAgent - ReAct 推理模式

结合推理和行动的循环模式。

```python
from app.agent.graph import create_react_agent

agent = create_react_agent(
    system_prompt="你是一个研究助手，善于使用工具收集信息"
)

response = await agent.get_response(
    message="查询最新的 AI 技术进展",
    session_id="session-123",
)
```

**流程图：**
```
    ┌──────────┐
    │  入口    │
    └────┬─────┘
         │
         ▼
   ┌─────────────┐
   │   Thought   │  思考需要做什么
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │   Action    │  决定执行什么工具
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Observation│  获取工具执行结果
   └──────┬──────┘
          │
          ┌──────┐
          │完成? │
          └──┬───┘
             │
       ┌─────┴─────┐
       │           │
      NO          YES
       │           │
       ▼           ▼
   ┌──────┐   ┌─────┐
   │Thought│   │ END │
   └──────┘   └─────┘
```

### 3. RouterGraph - 路由模式

根据用户意图路由到不同的子 Agent。

```python
from app.agent.multi_agent import create_router_agent

# 创建子 Agent
sales_agent = create_agent(system_prompt="你是销售专家")
support_agent = create_agent(system_prompt="你是技术支持")

# 创建路由 Agent
router = create_router_agent(
    agents={
        "sales": sales_agent,
        "support": support_agent,
    }
)

response = await router.ainvoke({
    "messages": [("user", "我想购买产品")],
})
```

**流程图：**
```
         ┌─────────┐
         │  用户   │
         └────┬────┘
              │
              ▼
    ┌─────────────────┐
    │   Router Agent  │  分析意图，决定路由
    └────────┬────────┘
             │
      ┌──────┼──────┬──────┐
      ▼      ▼      ▼      ▼
   ┌────┐ ┌────┐ ┌────┐ ┌────┐
   │销售│ │支持│ │... │ │默认│
   └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘
     │      │      │      │
     └──────┴──────┴──────┘
              │
              ▼
         ┌─────────┐
         │  回复   │
         └─────────┘
```

### 4. SupervisorGraph - 监督者模式

由监督者协调多个 Agent 协作完成任务。

```python
from app.agent.multi_agent import create_supervisor_agent

# 创建工作 Agent
researcher = create_agent(system_prompt="负责收集信息")
writer = create_agent(system_prompt="负责撰写报告")

# 创建监督者
supervisor = create_supervisor_agent(
    agents={
        "researcher": researcher,
        "writer": writer,
    },
    system_prompt="协调研究员和写作者完成任务"
)

response = await supervisor.ainvoke({
    "messages": [("user", "写一份关于 AI 的报告")],
})
```

**流程图：**
```
         ┌─────────┐
         │  用户   │
         └────┬────┘
              │
              ▼
    ┌─────────────────┐
    │  Supervisor     │  分配任务给 Worker
    └────────┬────────┘
             │
      ┌──────┼──────┬──────┐
      ▼      ▼      ▼      ▼
   ┌────┐ ┌────┐ ┌────┐ ┌────┐
   │研..│ │写..│ │... │ │... │  Workers 执行任务
   └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘
     │      │      │      │
     └──────┴──────┴──────┘
              │
              ▼
    ┌─────────────────┐
    │  Supervisor     │  评估结果，决定下一步
    └────────┬────────┘
             │
        ┌────┴────┐
        │         │
        ▼         ▼
    ┌───────┐ ┌─────┐
    │继续分配│ │完成 │
    └───────┘ └─────┘
```

### 5. SwarmGraph - 群体模式

多个 Agent 平等协作，共享状态。

```python
from app.agent.multi_agent import create_swarm_agent

workers = [
    create_agent(system_prompt=f"你是工作节点 {i+1}")
    for i in range(3)
]

swarm = create_swarm_agent(workers=workers)

response = await swarm.ainvoke({
    "messages": [("user", "分析这段文本的情感")],
})
```

---

## 工具系统

### 注册工具

使用装饰器自动注册工具：

```python
from app.agent.tools import tool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    """天气查询输入"""
    city: str = Field(..., description="城市名称")

@tool
async def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 实现天气查询逻辑
    return f"{city} 今天晴朗，温度 25°C"

# 工具自动注册到全局注册表
```

### 工具注册表

```python
from app.agent.tools import list_tools, get_tool

# 列出所有工具
tools = await list_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}")

# 获取特定工具
weather_tool = get_tool("get_weather")
if weather_tool:
    result = await weather_tool.ainvoke({"city": "北京"})
    print(result)
```

### 内置工具

| 工具名称 | 描述 | 参数 |
|----------|------|------|
| `calculator` | 数学计算 | expression: str |
| `search` | 网络搜索 | query: str |
| `get_weather` | 天气查询 | city: str |
| `database_query` | 数据库查询 | sql: str |

### 工具错误处理

```python
from app.agent.tools import set_tool_error_handler

def custom_handler(error: Exception) -> str:
    """自定义工具错误处理"""
    return f"工具执行失败: {error}"

set_tool_error_handler(custom_handler)
```

---

## 多 Agent 协作

### Router Agent - 路由模式

根据用户意图路由到不同的子 Agent。

```python
from app.agent.graph import compile_chat_graph
from app.agent.multi_agent import RouterAgent
from app.llm import LLMService

llm_service = LLMService()

# 创建子 Agent
sales_agent = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是销售专家，介绍产品功能和价格"
)

support_agent = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是客服专家，负责订单查询和售后处理"
)

# 创建路由 Agent
router = RouterAgent(
    llm_service=llm_service,
    agents={
        "Sales": sales_agent,
        "Support": support_agent,
    },
    router_prompt="根据用户意图选择 Sales(销售) 或 Support(客服)。"
)

graph = router.compile()

# 使用
response = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "我想买一个手机"}]},
    config={"configurable": {"thread_id": "test-1"}},
)
print(response["messages"][-1].content)
```

### Supervisor Agent - 监督者模式

由监督者协调多个 Worker Agent 完成复杂任务。

```python
from app.agent.graph import compile_chat_graph
from app.agent.multi_agent import SupervisorAgent
from app.llm import LLMService

llm_service = LLMService()

# 创建 Worker Agent
researcher = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是研究员，负责收集和分析信息"
)

writer = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是写手，负责整理和撰写报告"
)

reviewer = compile_chat_graph(
    llm_service=llm_service,
    system_prompt="你是审核员，负责审核报告质量"
)

# 创建监督 Agent
supervisor = SupervisorAgent(
    llm_service=llm_service,
    workers={
        "Researcher": researcher,
        "Writer": writer,
        "Reviewer": reviewer,
    },
    supervisor_prompt="协调 Researcher、Writer、Reviewer 完成报告撰写任务"
)

graph = supervisor.compile()

# 使用
response = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "写一份关于 AI 的报告"}]},
    config={"configurable": {"thread_id": "test-2"}},
)
```

### Handoff Agent - 交接模式

Agent 可以主动切换到其他 Agent，实现动态协作。

```python
from app.agent.multi_agent import HandoffAgent, create_swarm
from app.llm import LLMService
from langchain_core.tools import tool

llm_service = LLMService()

# 定义工具
@tool
async def search_products(query: str) -> str:
    """搜索产品信息"""
    return f"找到 3 个产品: {query}"

@tool
async def check_order_status(order_id: str) -> str:
    """查询订单状态"""
    return f"订单 {order_id} 状态: 已发货"

# 创建可切换的 Agent
alice = HandoffAgent(
    name="Alice",
    llm_service=llm_service,
    tools=[search_products],
    handoff_targets=["Bob"],
    system_prompt="你是 Alice，销售专家。遇到技术问题时转给 Bob。"
)

bob = HandoffAgent(
    name="Bob",
    llm_service=llm_service,
    tools=[check_order_status],
    handoff_targets=["Alice"],
    system_prompt="你是 Bob，客服专家。遇到销售咨询时转给 Alice。"
)

# 创建 Swarm
graph = create_swarm(
    agents=[alice, bob],
    default_agent="Alice",
)

# 使用
response = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "我要退款"}]},
    config={"configurable": {"thread_id": "test-3"}},
)
```

### 便捷函数 - 统一创建接口

使用 `create_multi_agent_system` 统一创建不同类型的多 Agent 系统：

```python
from app.agent.multi_agent import create_multi_agent_system
from app.agent.graph import compile_chat_graph
from app.llm import get_llm_service

llm_service = get_llm_service()

# Router 模式
router_graph = create_multi_agent_system(
    mode="router",
    llm_service=llm_service,
    agents={
        "Agent1": compile_chat_graph(llm_service=llm_service),
        "Agent2": compile_chat_graph(llm_service=llm_service),
    }
)

# Supervisor 模式
supervisor_graph = create_multi_agent_system(
    mode="supervisor",
    llm_service=llm_service,
    workers={
        "Worker1": compile_chat_graph(llm_service=llm_service),
        "Worker2": compile_chat_graph(llm_service=llm_service),
    }
)

# Swarm 模式
from app.agent.multi_agent import HandoffAgent

swarm_graph = create_multi_agent_system(
    mode="swarm",
    agents=[
        HandoffAgent(name="Agent1", llm_service=llm_service),
        HandoffAgent(name="Agent2", llm_service=llm_service),
    ],
    default_agent="Agent1"
)
```

### 完整示例

更多示例请参考 `app/agent/examples.py`：

```bash
# 查看示例代码
cat app/agent/examples.py

# 运行示例（需要设置 OPENAI_API_KEY）
python -m app.agent.examples
```

---

## 状态管理

### AgentState 定义

```python
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Agent 状态定义"""
    # 使用 add_messages reducer 自动处理消息追加
    messages: Annotated[list[BaseMessage], add_messages]

    # 用户标识
    user_id: str | None

    # 会话标识
    session_id: str

    # 防止无限循环
    iteration_count: int
    max_iterations: int
```

### 创建状态

```python
from app.agent.state import create_state_from_input

state = create_state_from_input(
    input_text="你好",
    user_id="user-123",
    session_id="session-abc",
)
```

### 状态滑动窗口

自动修剪旧消息，保持上下文在合理范围内：

```python
# 配置最大消息数
KIKI_AGENT_MAX_MESSAGES=100
```

---

## 检查点持久化

### PostgreSQL 检查点

```python
from app.agent import get_agent

# 使用 PostgreSQL 持久化检查点
agent = await get_agent(use_postgres_checkpointer=True)

# 所有对话状态自动持久化
await agent.get_response(
    message="你好",
    session_id="user-123",
)

# 可以随时恢复历史
history = await agent.get_chat_history("user-123")
```

### 时间旅行

获取历史状态：

```python
from langgraph.types import RunnableConfig

graph = get_cached_graph()

# 获取当前状态
config = RunnableConfig(configurable={"thread_id": "user-123"})
state = await graph.aget_state(config)

# 获取历史状态
states = []
async for state in graph.aget_state_history(config):
    states.append(state)
```

---

## 最佳实践

### 1. 系统提示词设计

```python
# 好的提示词
system_prompt = """
你是一个专业的技术顾问，擅长解答软件开发相关问题。

你的职责：
- 提供准确的技术建议
- 解释复杂概念时使用类比
- 当不确定时，诚实告知用户

沟通风格：
- 专业但友好
- 使用中文回答
- 避免使用过多术语
"""

# 避免过于简单
bad_prompt = "你是一个助手"
```

### 2. 错误处理

```python
from app.agent import create_agent
from app.core.config import get_settings

settings = get_settings()

agent = create_agent(
    system_prompt="你是一个有用的助手",
)

# 配置重试
agent._max_retries = settings.agent_max_retries
agent._retry_initial_interval = settings.agent_retry_initial_interval
```

### 3. 流式响应优化

```python
# 推荐：使用 astream 流式模式
async for chunk, metadata in graph.astream(
    input_data,
    config,
    stream_mode="messages",
):
    if hasattr(chunk, "content") and chunk.content:
        yield chunk.content
```

### 4. 成本优化

```python
from app.llm import get_llm_service

llm_service = get_llm_service()

# 简单任务使用便宜模型
cheap_llm = llm_service.get_llm_for_task("cost")

# 复杂任务使用强模型
smart_llm = llm_service.get_llm_for_task("quality")
```

### 5. 监控和追踪

```python
# 启用 Langfuse 追踪
KIKI_LANGCHAIN_TRACING_V2=true
KIKI_LANGCHAIN_API_KEY=your-key

# 查看追踪
# https://langfuse.com
```

---

## 调试技巧

### 1. 可视化图结构

```python
from app.agent.graph import compile_chat_graph

compiled = compile_chat_graph()

# 打印图结构
print(compiled.get_graph().print_ascii())
```

### 2. 查看状态变化

```python
async for chunk in graph.astream(
    input_data,
    config,
    stream_mode="values",
):
    print("状态更新:", chunk)
```

### 3. 追踪工具调用

```python
import structlog

logger = structlog.get_logger(__name__)

logger.info(
    "agent_call",
    session_id="user-123",
    message_count=len(state["messages"]),
    has_tool_calls=bool(tool_calls),
)
```

---

## 参考资源

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [LangChain 文档](https://python.langchain.com/)
- [项目架构文档](ARCHITECTURE.md)
- [API 文档](API.md)
