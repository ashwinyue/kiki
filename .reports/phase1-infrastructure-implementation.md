# Phase 1 基础设施实施报告

> 实施日期: 2025-02-04
> 参考: `.reports/multi-agent-architecture-improvement.md`

## 概述

完成了多 Agent 架构改进的 Phase 1：基础设施改进。包括分层 LLM 配置系统和统一 Agent 工厂。

## 实施的文件

### 1. 新建文件

| 文件 | 说明 |
|------|------|
| `app/agent/config/__init__.py` | Agent 配置模块入口 |
| `app/agent/config/llm_config.py` | 分层 LLM 配置核心实现 |
| `app/agent/graph/agent_factory.py` | 统一 Agent 工厂 |

### 2. 修改文件

| 文件 | 变更 |
|------|------|
| `app/agent/graph/__init__.py` | 导出 agent_factory 模块 |

## 核心功能

### 1. 分层 LLM 配置 (`app/agent/config/llm_config.py`)

#### LLM 类型定义

```python
LLMType = Literal[
    "reasoning",  # 推理模型（复杂规划、深度分析）
    "basic",      # 基础模型（通用对话、简单任务）
    "vision",     # 视觉模型（图像理解、多模态）
    "code",       # 代码模型（代码生成、代码分析）
]
```

#### Agent-LLM 映射

```python
AGENT_LLM_MAP = {
    # Supervisor 模式
    "supervisor": "reasoning",
    "router": "basic",

    # 专门化角色（DeerFlow 风格）
    "coordinator": "basic",
    "planner": "reasoning",
    "researcher": "basic",
    "analyst": "basic",
    "coder": "code",
    "reporter": "basic",

    # 通用角色
    "chat": "basic",
    "worker": "basic",
    "tools": "basic",
}
```

#### 分层配置

```python
LLM_CONFIG = {
    "reasoning": LLMTierConfig(
        model="deepseek-reasoner",
        provider="deepseek",
        max_tokens=100000,
    ),
    "basic": LLMTierConfig(
        model="gpt-4o",
        provider="openai",
    ),
    "vision": LLMTierConfig(
        model="gpt-4o",
        provider="openai",
    ),
    "code": LLMTierConfig(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        temperature=0.3,
    ),
}
```

#### 环境变量配置

支持通过环境变量覆盖默认配置：

```bash
# 覆盖 reasoning 模型
KIKI_LLM__REASONING__MODEL="gpt-4o"

# 覆盖 code 模型
KIKI_LLM__CODE__PROVIDER="openai"
KIKI_LLM__CODE__TEMPERATURE=0.1
```

### 2. 统一 Agent 工厂 (`app/agent/graph/agent_factory.py`)

#### 核心 API

```python
async def create_agent(
    agent_name: str,
    agent_type: str,
    tools: list[BaseTool] | None = None,
    prompt_template: str | None = None,
    system_prompt: str | None = None,
    interrupt_before_tools: list[str] | None = None,
    **llm_kwargs,
) -> CompiledStateGraph
```

#### 专门化角色创建函数

| 函数 | 说明 |
|------|------|
| `create_planner_agent()` | 创建 Planner Agent（规划任务） |
| `create_researcher_agent()` | 创建 Researcher Agent（信息检索） |
| `create_analyst_agent()` | 创建 Analyst Agent（数据分析） |
| `create_coder_agent()` | 创建 Coder Agent（代码生成） |
| `create_reporter_agent()` | 创建 Reporter Agent（报告生成） |

#### 批量创建

```python
async def create_agents(
    agent_configs: dict[str, dict[str, Any]],
) -> dict[str, CompiledStateGraph]
```

## 使用示例

### 基础用法

```python
from app.agent.config import get_llm_for_agent, AGENT_LLM_MAP
from app.agent.graph import create_agent, create_coder_agent

# 方式 1：通用创建函数
planner = await create_agent(
    agent_name="planner",
    agent_type="planner",
    system_prompt="你是一个任务规划专家",
)

# 方式 2：专门化角色函数
coder = await create_coder_agent(
    tools=[python_repl],
)

# 方式 3：直接获取 LLM
reasoning_llm = get_llm_for_agent("planner")
```

### 批量创建

```python
configs = {
    "planner": {
        "agent_type": "planner",
        "system_prompt": "规划任务",
    },
    "coder": {
        "agent_type": "coder",
        "tools": await alist_tools(),
    },
    "researcher": {
        "agent_type": "researcher",
        "tools": [tavily_search],
    },
}

agents = await create_agents(configs)
```

### Multi-Agent Graph 集成

```python
from app.agent.graph import MultiAgentGraphBuilder, create_agent

# 创建 workers
workers = {
    "planner": await create_agent("planner", "planner", tools=[]),
    "coder": await create_agent("coder", "coder", tools=tools),
}

# 构建图
builder = MultiAgentGraphBuilder(workers=workers)
graph = await builder.build_supervisor_graph()
```

## 设计决策

### 1. 为什么使用 `create_react_agent`？

| 优势 | 说明 |
|------|------|
| **标准化** | LangGraph 官方推荐模式 |
| **简洁** | 减少自定义代码 |
| **维护性** | 与 LangGraph 版本同步更新 |

### 2. 为什么需要分层 LLM？

| 收益 | 说明 |
|------|------|
| **成本优化** | 简单任务用便宜模型（DeepSeek） |
| **性能提升** | 复杂任务用强模型（GPT-4o/Claude） |
| **灵活配置** | 每种 Agent 类型可独立配置 |

### 3. 与现有代码的兼容性

- ✅ 不影响现有的 `ChatAgent` 实现
- ✅ 与 `LLMRegistry` 无缝集成
- ✅ 支持多提供商路由（`get_llm_for_task`）
- ✅ 支持 Checkpoint 持久化

## 测试

### 单元测试（待添加）

```python
# tests/unit/test_llm_config.py
async def test_get_llm_by_type():
    llm = get_llm_by_type("reasoning")
    assert llm is not None

async def test_get_llm_for_agent():
    llm = get_llm_for_agent("planner")
    assert llm is not None

# tests/unit/test_agent_factory.py
async def test_create_planner_agent():
    planner = await create_planner_agent()
    assert planner is not None
```

### 集成测试（待添加）

```python
async def test_multi_agent_with_factory():
    workers = {
        "planner": await create_planner_agent(),
        "coder": await create_coder_agent(tools=[python_repl]),
    }

    builder = MultiAgentGraphBuilder(workers=workers)
    graph = await builder.build_supervisor_graph()

    result = await graph.ainvoke(
        {"messages": ["创建一个排序算法"]},
        {"configurable": {"thread_id": "test-123"}}
    )
```

## 后续步骤

### Phase 2: 核心功能 (2-3 天)

```
1. 引入 Jinja2 提示词模板
   - app/agent/prompts/template.py
   - 创建模板文件目录
   - 更新 Agent 工厂使用模板

2. 定义专门化角色
   - app/agent/roles/ 目录
   - coordinator, planner, researcher, etc.
```

### Phase 3: 高级功能 (2-3 天)

```
3. 工具拦截器集成
   - 增强现有工具拦截器
   - 集成到 Agent 工厂

4. 智能路由
   - LLM 意图分类
   - 集成到 Supervisor 模式
```

## 风险和注意事项

| 风险 | 缓解措施 |
|------|----------|
| **LLM 配置复杂** | 提供默认配置，文档完善 |
| **兼容性问题** | 保留旧接口，渐进迁移 |
| **性能影响** | LLM 缓存，异步处理 |

## 参考

- DeerFlow: https://github.com/bytedance/deer-flow
- LangGraph Prebuilt: https://langchain-ai.github.io/langgraph/reference/prebuilt/
- 多 Agent 架构改进建议: `.reports/multi-agent-architecture-improvement.md`
