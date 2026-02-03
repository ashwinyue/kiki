# Kiki 多 Agent 架构改进建议

> 基于当前项目架构 + DeerFlow 参考设计
> 分析日期: 2025-02-03

## 一、当前架构分析

### 1.1 现有架构概览

```
Kiki Multi-Agent 架构：

┌─────────────────────────────────────────────────────────────┐
│                    MultiAgent (基类)                        │
├─────────────────────────────────────────────────────────────┤
│  SupervisorAgent          │          RouterAgent             │
│  ┌──────────────┐         │          ┌──────────┐          │
│  │ Supervisor   │         │          │  Router  │          │
│  │    ↓         │         │          │    ↓     │          │
│  └──────────────┘         │          └──────────┘          │
│       ↓                    │              ↓                  │
│  [Worker_A, Worker_B, Worker_C]  →  [Worker_A, Worker_B] → END │
│       ↓                    │                                   │
│  → Supervisor → END       │                                   │
└─────────────────────────────────────────────────────────────┘

关键组件：
- app/agent/multi_agent.py - MultiAgent, SupervisorAgent, RouterAgent
- app/agent/graph/multi_agent.py - MultiAgentGraphBuilder, supervisor_node
- app/agent/graph/builder.py - build_chat_graph, compile_chat_graph
- app/agent/graph/nodes.py - chat_node
```

### 1.2 当前优势

| 方面 | 优势 |
|------|------|
| **架构模式** | 支持 Supervisor 和 Router 两种模式，适应不同场景 |
| **调用追踪** | 集成 AgentExecutionTracker，完整的执行链追踪 |
| **状态管理** | 统一的 MultiAgentState，支持迭代控制 |
| **Checkpointer** | 支持 PostgreSQL 持久化 |
| **租户隔离** | tenant_id 隔离，支持多租户 |

### 1.3 当前限制

| 方面 | 限制 | 改进空间 |
|------|------|----------|
| **LLM 配置** | 单一 LLM 配置，无角色区分 | ⭐⭐⭐⭐⭐ 引入分层 LLM |
| **Agent 创建** | ChatAgent 实例化，无统一工厂 | ⭐⭐⭐⭐ 使用 create_react_agent |
| **路由逻辑** | 基于关键词的简单匹配 | ⭐⭐⭐⭐ LLM 意图分类 |
| **提示词** | 硬编码字符串 | ⭐⭐⭐⭐⭐ Jinja2 模板 |
| **工具系统** | 基础工具绑定 | ⭐⭐⭐⭐ 工具拦截器 |
| **Agent 角色** | 通用 Worker，无角色区分 | ⭐⭐⭐⭐⭐ 专门化角色定义 |

---

## 二、DeerFlow 架构参考

### 2.1 DeerFlow 架构

```
DeerFlow 深度研究架构：

┌─────────────────────────────────────────────────────────────┐
│                      Coordinator                          │
│                  (入口 + 背景调查)                        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Planner                                │
│              (任务分解 + 计划生成)                          │
│  • 分析目标                                                 │
│  • 创建结构化计划                                             │
│  • 决定迭代次数                                               │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 Research Team                             │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ Researcher   │  │   Analyst    │  │    Coder     │  │
│  │ (Web Search) │  │ (Data Analysis)│  │ (Python REPL) │  │
│  └──────────────┘  └─────────────┘  └──────────────┘  │
│         ↓              ↓                ↓               │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Reporter                                │
│               (聚合 + 生成报告)                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 DeerFlow 核心设计

#### 1. 分层 LLM 配置

```python
# deer-flow/src/config/agents.py
AGENT_LLM_MAP = {
    "coordinator": "basic",
    "planner": "reasoning",
    "researcher": "basic",
    "analyst": "basic",
    "coder": "code",
    "reporter": "basic",
}

# deer-flow/src/llms/llm.py
def get_llm_by_type(llm_type: LLMType) -> BaseChatModel:
    # 分层 LLM + 配置合并 + 缓存
```

#### 2. 统一 Agent 创建

```python
# deer-flow/src/agents/agents.py
def create_agent(
    agent_name: str,
    agent_type: str,      # 决定 LLM 类型
    tools: list,
    prompt_template: str,   # 提示词模板名称
    interrupt_before_tools: list[str] = None,
) -> CompiledStateGraph:
    # 使用 create_react_agent
    agent = create_react_agent(
        name=agent_name,
        model=get_llm_by_type(llm_type),
        tools=processed_tools,
        prompt=lambda state: apply_prompt_template(prompt_template, state),
    )
```

#### 3. 计划驱动路由

```python
# deer-flow/src/graph/builder.py
def continue_to_running_research_team(state: State):
    """根据当前计划状态路由"""
    current_plan = state.get("current_plan")

    # 找到第一个未完成的步骤
    for step in current_plan.steps:
        if not step.execution_res:
            if step.step_type == StepType.RESEARCH:
                return "researcher"
            elif step.step_type == StepType.ANALYSIS:
                return "analyst"
            elif step.step_type == StepType.PROCESSING:
                return "coder"

    return "planner"
```

#### 4. Jinja2 提示词模板

```python
# deer-flow/src/prompts/template.py
def apply_prompt_template(
    prompt_name: str,
    state: AgentState,
    locale: str = "en-US"
) -> list:
    """应用提示词模板"""
    # 支持 locale 切换
    # 支持变量注入
    # 自动添加系统消息
```

#### 5. 工具拦截器

```python
# deer-flow/src/agents/tool_interceptor.py
class ToolInterceptor:
    """工具执行拦截 - Human-in-the-Loop"""

    def wrap_tools(tools: list, interrupt_before_tools: list[str]) -> list:
        """包装工具，添加拦截逻辑"""
        # 在执行前检查是否需要用户批准
```

---

## 三、改进方案

### 3.1 引入分层 LLM 配置 ⭐⭐⭐⭐⭐

**当前问题**：Kiki 使用单一 LLM 配置，所有 Agent 使用相同的模型。

**改进方案**：

```python
# app/agent/config/llm_config.py (新建)
from enum import Literal
from typing import Literal

# LLM 类型定义
LLMType = Literal[
    "reasoning",  # 推理模型（复杂规划）
    "basic",      # 基础模型（通用对话）
    "vision",     # 视觉模型（图像理解）
    "code",       # 代码模型（代码生成）
]

# Agent-LLM 映射
AGENT_LLM_MAP: dict[str, LLMType] = {
    # Supervisor 模式
    "supervisor": "reasoning",
    "router": "basic",

    # 专门化角色
    "planner": "reasoning",
    "researcher": "basic",
    "analyst": "basic",
    "coder": "code",
    "reporter": "basic",

    # 通用角色
    "chat": "basic",
}

# 分层配置
LLM_CONFIG = {
    "reasoning": {
        "model": "deepseek-reasoner",
        "api_key": "$DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "max_tokens": 100000,
    },
    "basic": {
        "model": "gpt-4o",
        "api_key": "$OPENAI_API_KEY",
        "temperature": 0.7,
    },
    "code": {
        "model": "claude-3.5-sonnet",
        "api_key": "$ANTHROPIC_API_KEY",
    },
}
```

### 3.2 使用 create_react_agent 模式 ⭐⭐⭐⭐⭐

**当前问题**：Kiki 使用 ChatAgent 实例化 Worker，每次创建新实例。

**改进方案**：

```python
# app/agent/graph/agent_factory.py (新建)
from langgraph.prebuilt import create_react_agent
from app.agent.config.llm_config import AGENT_LLM_MAP, get_llm_by_type

def create_agent(
    agent_name: str,
    agent_type: str,
    tools: list,
    prompt_template: str,
    interrupt_before_tools: list[str] | None = None,
) -> CompiledStateGraph:
    """统一 Agent 创建工厂

    Args:
        agent_name: Agent 名称
        agent_type: Agent 类型（决定 LLM 类型）
        tools: 工具列表
        prompt_template: 提示词模板名称
        interrupt_before_tools: 需要拦截的工具列表
    """
    # 1. 获取 LLM 类型
    llm_type = AGENT_LLM_MAP.get(agent_type, "basic")
    llm = get_llm_by_type(llm_type)

    # 2. 工具拦截器
    processed_tools = tools
    if interrupt_before_tools:
        from app.agent.tools.interceptor import wrap_tools_with_interceptor
        processed_tools = wrap_tools_with_interceptor(
            tools,
            interrupt_before_tools
        )

    # 3. 创建 ReAct Agent
    agent = create_react_agent(
        name=agent_name,
        model=llm,
        tools=processed_tools,
        prompt=lambda state: apply_prompt_template(
            prompt_template,
            state,
            locale=state.get("locale", "zh-CN")
        ),
    )

    logger.info("react_agent_created", agent_name=agent_name, agent_type=agent_type)
    return agent
```

### 3.3 改进条件边路由 ⭐⭐⭐⭐

**当前问题**：基于关键词的简单路由，不够智能。

**改进方案**：

```python
# app/agent/graph/routing.py (增强)
from langchain_openai import ChatOpenAI
from app.config.settings import get_settings

def create_intent_classifier():
    """创建意图分类器（使用 LLM）"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=get_settings().openai_api_key,
    )

async def llm_based_router(
    messages: list[BaseMessage],
    allowed_workers: list[str],
    llm: ChatOpenAI | None = None,
) -> str:
    """基于 LLM 的智能路由

    Args:
        messages: 消息列表
        allowed_workers: 允许的 worker 列表

    Returns:
        选中的 agent_id
    """
    llm = llm or create_intent_classifier()

    # 构建分类提示
    worker_descriptions = "\n".join([
        f"- {agent_id}: {agent_config.get('description', agent_id)}"
        for agent_id, agent_config in allowed_workers.items()
    ])

    prompt = f"""分析用户意图，选择最合适的助手：

可用助手：
{worker_descriptions}

用户消息：{messages[-1].content}

只返回助手名称，不要其他内容。"""

    try:
        response = await llm.ainvoke(prompt)
        selected = response.content.strip()

        # 验证返回的 agent_id 是否有效
        if selected in allowed_workers:
            return selected

        logger.warning("llm_router_returned_invalid_agent", selected=selected)
        return list(allowed_workers.keys())[0]

    except Exception as e:
        logger.error("llm_router_failed", error=str(e))
        return list(allowed_workers.keys())[0]
```

### 3.4 引入 Jinja2 提示词模板 ⭐⭐⭐⭐⭐

**当前问题**：提示词是硬编码字符串，难以维护。

**改进方案**：

```python
# app/agent/prompts/template.py (重构)
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from datetime import UTC, datetime

# 初始化 Jinja2 环境
TEMPLATES_DIR = Path(__file__).parent
env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

def render_prompt(
    template_name: str,
    state: dict,
    locale: str = "zh-CN",
) -> str:
    """渲染提示词模板

    Args:
        template_name: 模板名称（不含扩展名）
        state: 状态变量
        locale: 语言区域（如 zh-CN, en-US）

    Returns:
        渲染后的提示词
    """
    # 准备变量
    variables = {
        "CURRENT_TIME": datetime.now(UTC).strftime("%a %b %d %H:%M:%S %z"),
        **state,
    }

    # 尝试本地化模板
    normalized_locale = locale.replace("-", "_")
    try:
        template = env.get_template(f"{template_name}.{normalized_locale}.md")
    except:
        template = env.get_template(f"{template_name}.md")

    return template.render(**variables)

# 模板文件结构
# app/agent/prompts/
# ├── supervisor.zh_CN.md
# ├── supervisor.md
# ├── researcher.zh_CN.md
# ├── researcher.md
# ├── coder.zh_CN.md
# └── coder.md
```

### 3.5 定义专门化 Agent 角色 ⭐⭐⭐⭐⭐

**当前问题**：Worker 是通用角色，无明确职责区分。

**改进方案**：

```python
# app/agent/roles/ (新建目录结构)
# app/agent/roles/
# ├── __init__.py
# ├── coordinator.py
# ├── planner.py
# ├── researcher.py
# ├── analyst.py
# ├── coder.py
# └── reporter.py

# app/agent/roles/planner.py (示例)
from langgraph.graph import StateGraph
from app.agent.graph.agent_factory import create_agent

class PlannerRole:
    """Planner 角色 - 任务规划和分解"""

    def __init__(self):
        self.role_name = "planner"
        self.agent_type = "reasoning"
        self.tools = []  # Planner 通常不需要工具

    def create_node(self, state_cls, config_class):
        """创建 Planner 节点"""
        async def planner_node(state: state_cls, config):
            """Planner 节点：创建和更新计划"""
            messages = state.get("messages", [])

            # 分析目标
            goal = messages[-1].content if messages else ""

            # 检查是否需要更多信息
            background_results = state.get("background_investigation_results")

            # 创建或更新计划
            current_plan = state.get("current_plan")
            if current_plan and self._is_plan_complete(current_plan):
                return {"goto": "reporter"}

            # 生成新计划
            new_plan = await self._generate_plan(goal, background_results)

            return {
                "current_plan": new_plan,
                "plan_iterations": state.get("plan_iterations", 0) + 1,
            }

        return planner_node

    async def _generate_plan(self, goal: str, context: str) -> dict:
        """生成结构化计划"""
        # 使用 reasoning LLM 生成计划
        return {
            "goal": goal,
            "steps": [
                {"step_type": "research", "description": "搜索相关信息"},
                {"step_type": "analysis", "description": "分析数据"},
            ],
        }
```

### 3.6 工具拦截器集成 ⭐⭐⭐⭐

**当前问题**：工具执行无拦截机制。

**改进方案**：

```python
# app/agent/tools/interceptor.py (增强现有实现)
from typing import list
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

class ToolInterceptor:
    """工具拦截器 - Human-in-the-Loop"""

    def wrap_tools(
        self,
        tools: list[BaseTool],
        interrupt_before_tools: list[str]
    ) -> list[BaseTool]:
        """包装工具，添加拦截逻辑"""
        wrapped_tools = []

        for tool in tools:
            if tool.name in interrupt_before_tools:
                wrapped_tools.append(self._wrap_tool(tool))
            else:
                wrapped_tools.append(tool)

        return wrapped_tools

    def _wrap_tool(self, tool: BaseTool) -> BaseTool:
        """包装单个工具"""
        original_func = tool.func

        async def intercepted_func(*args, **kwargs):
            tool_name = tool.name

            # 触发中断
            feedback = interrupt(
                f"即将执行工具: {tool_name}\n"
                f"输入: {args[0] if args else kwargs}\n"
                f"是否批准？"
            )

            # 检查审批结果
            if not self._is_approved(feedback):
                return {
                    "error": "工具执行被用户拒绝",
                    "tool": tool_name,
                    "status": "rejected",
                }

            # 执行原始工具
            return await original_func(*args, **kwargs)

        object.__setattr__(tool, "func", intercepted_func)
        return tool

    @staticmethod
    def _is_approved(feedback: str) -> bool:
        """解析审批结果"""
        approved_keywords = ["yes", "ok", "approved", "proceed", "continue"]
        return any(
            keyword in feedback.lower()
            for keyword in approved_keywords
        )
```

---

## 四、实施路线图

### Phase 1: 基础设施 (1-2 天)

```
1. 创建分层 LLM 配置
   - app/agent/config/llm_config.py
   - app/agent/config/agents.py
   - 更新 app/llm/service.py

2. 创建 Agent 工厂
   - app/agent/graph/agent_factory.py
   - 使用 create_react_agent 模式
```

### Phase 2: 核心功能 (2-3 天)

```
3. 引入 Jinja2 提示词模板
   - app/agent/prompts/template.py
   - 创建模板文件目录
   - 更新现有 Agent 使用模板

4. 定义专门化角色
   - app/agent/roles/ 目录
   - coordinator, planner, researcher, etc.
```

### Phase 3: 高级功能 (2-3 天)

```
5. 工具拦截器集成
   - 增强现有工具拦截器
   - 集成到 Agent 工厂

6. 智能路由
   - LLM 意图分类
   - 集成到 Supervisor 模式
```

### Phase 4: 优化和测试 (2-3 天)

```
7. 端到端测试
8. 性能优化
9. 文档更新
```

---

## 五、具体代码示例

### 5.1 分层 LLM 使用示例

```python
# 使用新的分层 LLM 配置
from app.agent.config.llm_config import AGENT_LLM_MAP, get_llm_by_type

# Planner 使用推理模型
llm_type = AGENT_LLM_MAP["planner"]  # "reasoning"
planner_llm = get_llm_by_type(llm_type)

# Coder 使用代码模型
coder_llm = get_llm_by_type(AGENT_LLM_MAP["coder"])  # "code"
```

### 5.2 新 Agent 创建模式

```python
from app.agent.graph.agent_factory import create_agent
from app.agent.tools import alist_tools

async def create_planner_agent():
    tools = await alist_tools()  # Planner 通常不需要工具
    return create_agent(
        agent_name="planner",
        agent_type="planner",
        tools=tools,
        prompt_template="planner",
    )
```

### 5.3 智能路由使用

```python
from app.agent.graph.routing import llm_based_router

# 在 Supervisor 模式中使用
async def supervisor_node(state, config):
    messages = state.get("messages", [])
    allowed_workers = list(state.get("workers", {}).keys())

    # 使用 LLM 智能路由
    next_agent = await llm_based_router(messages, allowed_workers)

    return Command(goto=(next_agent,))
```

---

## 六、迁移建议

### 兼容性考虑

1. **向后兼容**：保留现有的 SupervisorAgent 和 RouterAgent 接口
2. **渐进迁移**：先在新功能中测试，再逐步迁移
3. **配置切换**：通过配置控制新旧模式

### 迁移步骤

1. **创建新模块**（不影响现有代码）
2. **并行测试**（新旧系统对比）
3. **逐步切换**（从非关键功能开始）
4. **移除旧代码**（确认稳定后）

---

## 七、预期收益

| 改进项 | 收益 | 优先级 |
|--------|------|--------|
| 分层 LLM | 降低成本、提高质量 | P0 |
| create_react_agent | 统一模式、减少代码 | P0 |
| Jinja2 模板 | 可维护性、国际化 | P1 |
| 智能路由 | 准确性提升 | P1 |
| 专门化角色 | 功能增强 | P2 |
| 工具拦截器 | 安全性提升 | P2 |

---

## 八、风险和注意事项

| 风险 | 缓解措施 |
|------|----------|
| **复杂度增加** | 分阶段实施，充分测试 |
| **性能影响** | LLM 缓存，异步处理 |
| **配置复杂** | 提供默认配置，文档完善 |
| **兼容性** | 保留旧接口，渐进迁移 |
