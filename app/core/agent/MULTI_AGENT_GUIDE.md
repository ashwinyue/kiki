# 多 Agent 架构指南

## 概述

本项目支持三种主流的多 Agent 协作模式，每种模式适用于不同的场景。所有模式都充分利用了 LangChain 和 LangGraph 的内置功能：

- **ChatPromptTemplate** - 提示词模板管理
- **with_structured_output** - 类型安全的结构化输出
- **add_conditional_edges** - 声明式路由
- **LCEL Chain** - 链式调用

## 模式对比

| 模式 | 适用场景 | 复杂度 | 灵活性 | 示例 |
|------|----------|--------|--------|------|
| **Router** | 意图分类、任务分发 | ⭐⭐ | ⭐⭐⭐ | 客服路由（销售/技术/账单） |
| **Supervisor** | 复杂任务分解 | ⭐⭐⭐ | ⭐⭐ | 报告生成（研究→撰写→审核） |
| **Swarm/Handoff** | 动态协作 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 多角色对话（销售↔技术） |

---

## 1. Router Agent（路由模式）

### 架构图

```
        User Request
             │
             ▼
    ┌──────────────────┐
    │   Router Agent   │  ← 意图识别 (structured_output)
    │   (LLM 分类)      │
    └────────┬─────────┘
             │
      ┌──────┼──────┬──────┐
      ▼      ▼      ▼      ▼
   Sales  Support  Billing  General
```

### 特性

- ✅ 使用 `with_structured_output(RouteDecision)` 确保类型安全
- ✅ 返回置信度和原因，便于调试
- ✅ 自动回退到默认 Agent

### 代码示例

```python
from app.core.agent.multi_agent import RouterAgent, create_multi_agent_system
from app.core.agent.graph import AgentGraph
from app.core.llm import get_llm_service

llm_service = get_llm_service()

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

# 使用
from langchain_core.messages import HumanMessage

response = await router_system.ainvoke(
    {"messages": [HumanMessage(content="我要买手机")]},
    config={"configurable": {"thread_id": "session-1"}},
)
```

### 路由决策结构

```python
class RouteDecision(BaseModel):
    agent: str = Field(description="目标 agent 名称")
    reason: str = Field(description="选择原因")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
```

---

## 2. Supervisor Agent（监督模式）

### 架构图

```
                    User Task
                       │
                       ▼
              ┌─────────────────┐
              │  Supervisor     │  ← 任务分解 (structured_output)
              │  (规划 + 分配)   │
              └────────┬─────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Worker1 │   │ Worker2 │   │ Worker3 │
    │ (研究)  │   │ (撰写)  │   │ (审核)  │
    └────┬────┘   └────┬────┘   └────┬────┘
         │             │             │
         └─────────────┴─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Supervisor     │  ← 汇总结果
              └─────────────────┘
```

### 特性

- ✅ 使用 `with_structured_output(SupervisorDecision)` 管理任务流程
- ✅ 支持多轮任务分配
- ✅ 自动汇总 Worker 结果

### 代码示例

```python
from app.core.agent.multi_agent import SupervisorAgent
from app.core.agent.graph import AgentGraph

# 创建 Worker Agent
researcher = AgentGraph(llm_service, system_prompt="研究员...")
writer = AgentGraph(llm_service, system_prompt="写手...")
reviewer = AgentGraph(llm_service, system_prompt="审核员...")

# 创建监督系统
supervisor_system = SupervisorAgent(
    llm_service=llm_service,
    workers={
        "Researcher": researcher,
        "Writer": writer,
        "Reviewer": reviewer,
    },
)

graph = supervisor_system.compile()

from langchain_core.messages import HumanMessage

response = await graph.ainvoke(
    {"messages": [HumanMessage(content="写一份 AI 行业报告")]},
    config={"configurable": {"thread_id": "task-1"}},
)
```

### 监督决策结构

```python
class SupervisorDecision(BaseModel):
    next: str = Field(description="下一个 worker 名称，或 'END'")
    status: Literal["working", "done"] = Field(description="任务状态")
    message: str = Field(default="", description="给用户或 worker 的消息")
```

---

## 3. Handoff Agent（Swarm 模式）

### 架构图

```
    ┌─────────┐              ┌─────────┐
    │ Alice   │◄─handoff────│  Bob    │
    │ (销售)  │──handoff───▶│ (技术)  │
    └─────────┘              └─────────┘
         ▲                        │
         │                        │
         └────────────────────────┘
              动态切换 (工具调用)
```

### 特性

- ✅ Agent 通过工具调用主动发起切换
- ✅ 支持双向切换
- ✅ LLM 自主决定何时切换

### 代码示例

```python
from app.core.agent.multi_agent import HandoffAgent, create_swarm

# 创建可切换 Agent
alice = HandoffAgent(
    name="Alice",
    llm_service=llm_service,
    tools=[search_products],
    handoff_targets=["Bob"],  # 可以切换到 Bob
    system_prompt="销售专家...",
)

bob = HandoffAgent(
    name="Bob",
    llm_service=llm_service,
    tools=[check_specifications],
    handoff_targets=["Alice"],  # 可以切换到 Alice
    system_prompt="技术专家...",
)

# 创建 Swarm
swarm = create_swarm(
    agents=[alice, bob],
    default_agent="Alice",
)

# 使用 - Alice 遇到技术问题会自动转给 Bob
from langchain_core.messages import HumanMessage

response = await swarm.ainvoke(
    {"messages": [HumanMessage(content="这个产品的技术参数是什么？")]},
    config={"configurable": {"thread_id": "chat-1"}},
)
```

---

## 选择建议

### 选择 Router Agent 当：

- ✅ 用户意图清晰可分类
- ✅ 各 Agent 独立工作，无需协作
- ✅ 需要快速响应

### 选择 Supervisor Agent 当：

- ✅ 任务需要多个步骤
- ✅ 需要协调多个 Worker
- ✅ 需要汇总最终结果

### 选择 Handoff Agent 当：

- ✅ 对话流程不确定
- ✅ Agent 需要自主决定切换
- ✅ 需要灵活的协作模式

---

## 高级特性

### 自定义结构化输出

你可以定义自己的决策模型：

```python
from pydantic import BaseModel, Field
from app.core.llm import get_llm_service

class CustomDecision(BaseModel):
    action: str = Field(description="要执行的操作")
    params: dict = Field(default_factory=dict, description="参数")

llm_service = get_llm_service()
structured_llm = llm_service.with_structured_output(CustomDecision)

decision: CustomDecision = await structured_llm.ainvoke(messages)
```

### 使用 ChatPromptTemplate

所有 Agent 都支持自定义提示词模板：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，专注于{domain}。"),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}"),
])

# 与 LLM 链接
chain = prompt | llm
```

### 条件路由

使用 `add_conditional_edges` 实现复杂路由：

```python
from typing import Literal
from langgraph.graph import StateGraph, END

def route_by_condition(state: AgentState) -> Literal["a", "b", "__end__"]:
    value = state.get("condition")
    if value == "high":
        return "a"
    elif value == "low":
        return "b"
    return "__end__"

builder = StateGraph(AgentState)
builder.add_conditional_edges(
    "decision",
    route_by_condition,
    {
        "a": "process_a",
        "b": "process_b",
        "__end__": END,
    }
)
```

---

## 参考

- [LangGraph Multi-Agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [LangGraph Structured Output](https://python.langchain.com/docs/how_to/structured_output/)
- [LangGraph Swarm](https://github.com/langchain-ai/langgraph-swarm)
