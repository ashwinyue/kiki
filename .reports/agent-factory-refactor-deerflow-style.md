# Agent 工厂重构报告（DeerFlow 风格）

> 重构日期: 2025-02-04
> 参考: DeerFlow 的设计理念

## 重构目标

将 Agent 创建统一为 DeerFlow 风格：**所有 Agent 都是 `CompiledStateGraph`，不需要额外的类包装**。

## 重构前后对比

### 重构前（混合模式）

```python
# 问题：多种 Agent 创建方式，不统一
chat_agent = ChatAgent(system_prompt="...")        # 使用 compile_chat_graph
supervisor = SupervisorAgent(workers={...})     # 使用 MultiAgentGraphBuilder
planner = await create_planner_agent(...)        # 专门化函数
```

**问题：**
1. `ChatAgent` 类继承 `BaseAgent`，增加复杂度
2. 专门化函数（create_planner_agent）返回 BaseAgent
3. 新工厂返回 CompiledStateGraph，接口不统一

### 重构后（DeerFlow 风格）

```python
# 统一：所有 Agent 都是 CompiledStateGraph
from app.agent.graph import create_agent, AGENT_REGISTRY

# 方式 1：直接创建
planner = create_agent(
    agent_name="planner",
    agent_type="planner",
    tools=[],
    prompt_template="planner",
)

# 方式 2：通过配置创建
config = AGENT_REGISTRY["planner"]
planner = create_agent(
    agent_name="planner",
    agent_type=config["agent_type"],
    tools=config["tools"](),
    prompt_template=config["prompt_template"],
)
```

## 核心设计

### 1. 简化的 Agent 工厂 (`agent_factory.py`)

```python
def create_agent(
    agent_name: str,
    agent_type: str,
    tools: list[BaseTool] | None = None,
    prompt_template: str | None = None,
    interrupt_before_tools: list[str] | None = None,
) -> CompiledStateGraph:
    """统一 Agent 创建工厂（同步函数）

    - 使用 AGENT_LLM_MAP 选择 LLM 类型
    - 使用 Jinja2 模板渲染提示词
    - 返回 CompiledStateGraph（create_react_agent）
    """
```

**关键特点：**
- **同步函数**：不需要 async，直接返回图
- **配置驱动**：通过 AGENT_LLM_MAP 和 prompt_template 配置
- **无类包装**：直接返回 LangGraph 的 CompiledStateGraph

### 2. Agent 配置文件 (`agents.py`)

```python
AGENT_REGISTRY = {
    "planner": {
        "agent_type": "planner",
        "prompt_template": "planner",
        "description": "规划者 - 任务分解和计划生成",
        "tools": lambda: [],  # 延迟加载
    },
    "researcher": {
        "agent_type": "researcher",
        "prompt_template": "researcher",
        "description": "研究员 - 信息检索和验证",
        "tools": get_research_tools,  # 函数引用
    },
    # ... 更多 agents
}

# 预配置 Agent 组
RESEARCH_TEAM = ["planner", "researcher", "analyst", "coder", "reporter"]
SUPERVISOR_TEAM = ["supervisor", "researcher", "coder"]
```

### 3. 与现有系统的兼容性

**现有 MultiAgent 系统如何使用新工厂：**

```python
from app.agent.graph import create_agent, MultiAgentGraphBuilder

# 创建 Agent 图（使用新工厂）
planner_graph = create_agent(
    agent_name="planner",
    agent_type="planner",
    prompt_template="planner",
)

# 包装为节点函数（兼容 MultiAgentGraphBuilder）
def planner_node(state, config):
    # 调用 planner_graph
    result = await planner_graph.ainvoke(
        {"messages": state["messages"]},
        config,
    )
    return result

# 添加到 MultiAgent
builder = MultiAgentGraphBuilder()
builder.add_node("planner", planner_node)
```

## 文件变更

| 文件 | 状态 | 说明 |
|------|------|------|
| `agent_factory.py` | 重写 | 简化为 DeerFlow 风格 |
| `agents.py` | 新建 | Agent 配置注册表 |
| `integration.py` | 删除 | 不再需要混合模式 |
| `__init__.py` | 更新 | 导出新接口 |

## 设计优势

| 方面 | 优势 |
|------|------|
| **简洁性** | 一个函数创建所有 Agent |
| **一致性** | 所有 Agent 都是 CompiledStateGraph |
| **配置驱动** | 集中配置，易于管理 |
| **兼容性** | 与 LangGraph 生态无缝集成 |
| **扩展性** | 新增 Agent 只需添加配置 |

## 使用示例

### 基础使用

```python
from app.agent.graph import create_agent

# 创建 Planner
planner = create_agent(
    agent_name="planner",
    agent_type="planner",
    tools=[],
    prompt_template="planner",
)

# 调用 Agent
result = await planner.ainvoke(
    {"messages": [("user", "创建排序算法")]},
    {"configurable": {"thread_id": "session-123"}},
)
```

### 批量创建

```python
from app.agent.graph import AGENT_REGISTRY, create_agent

agents = {}
for agent_id, config in AGENT_REGISTRY.items():
    agents[agent_id] = create_agent(
        agent_name=agent_id,
        agent_type=config["agent_type"],
        tools=config["tools"](),
        prompt_template=config["prompt_template"],
    )
```

### 预配置团队

```python
from app.agent.graph import RESEARCH_TEAM, create_agent

# 创建研究团队
research_team = {}
for agent_id in RESEARCH_TEAM:
    config = AGENT_REGISTRY[agent_id]
    research_team[agent_id] = create_agent(
        agent_name=agent_id,
        **{k: v() if callable(k) else v for k, v in config.items()
         if k != "description"},
    )
```

## 后续工作

| 任务 | 说明 |
|------|------|
| **更新 multi_agent.py** | 使用新工厂创建 worker 节点 |
| **创建 workflow.py** | 参考 DeerFlow 的编排模式 |
| **工具拦截器** | Phase 3 高级功能 |
| **测试** | 单元测试和集成测试 |

## 参考

- DeerFlow agents.py: `aold/deer-flow/src/agents/agents.py`
- DeerFlow config: `aold/deer-flow/src/config/agents.py`
