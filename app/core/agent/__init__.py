"""Agent 核心模块

包含：
- state: Agent 状态定义（使用 add_messages reducer）
- graphs: LangGraph 工作流（BaseGraph + ChatGraph）
- tools: 工具系统（registry + builtin 示例工具）
- agent: Agent 管理类（LangGraphAgent 门面）
- multi_agent: 多 Agent 系统（Router/Supervisor/Swarm）
- checkpoint: 检查点管理（PostgreSQL）

目录结构:
    agent/
    ├── __init__.py       # 本文件
    ├── state.py          # AgentState 定义
    ├── graphs/           # 图工作流
    │   ├── base.py       # BaseGraph 抽象基类
    │   ├── chat.py       # ChatGraph 基础对话图
    │   ├── nodes.py      # 节点函数
    │   └── routes.py     # 路由函数
    ├── tools/            # 工具系统
    │   ├── registry.py   # 工具注册表
    │   └── builtin/      # 内置示例工具
    ├── agent.py          # LangGraphAgent 门面类
    └── multi_agent.py    # 多 Agent 系统
"""

from app.core.agent.agent import (
    LangGraphAgent,
    create_agent,
    get_agent,
)
from app.core.agent.graphs import (
    BaseGraph,
    ChatGraph,
    chat_node,
    create_chat_graph,
    route_by_tools,
    tools_node,
)
from app.core.agent.multi_agent import (
    HandoffAgent,
    RouterAgent,
    SupervisorAgent,
    create_handoff_tool,
    create_multi_agent_system,
    create_swarm,
)
from app.core.agent.state import (
    AgentState,
    create_initial_state,
    create_state_from_input,
)
from app.core.agent.tools import (
    calculate,
    get_tool,
    get_tool_node,
    get_weather,
    list_tools,
    register_tool,
    search_database,
)

__all__ = [
    # State
    "AgentState",
    "create_initial_state",
    "create_state_from_input",
    # Graphs - 抽象基类
    "BaseGraph",
    # Graphs - 具体实现
    "ChatGraph",
    "create_chat_graph",
    # Graphs - 节点和路由（供扩展使用）
    "chat_node",
    "tools_node",
    "route_by_tools",
    # Tools - 注册系统
    "register_tool",
    "get_tool",
    "list_tools",
    "get_tool_node",
    # Tools - 内置示例
    "search_database",
    "get_weather",
    "calculate",
    # Agent
    "LangGraphAgent",
    "get_agent",
    "create_agent",
    # Multi-Agent
    "RouterAgent",
    "SupervisorAgent",
    "HandoffAgent",
    "create_handoff_tool",
    "create_swarm",
    "create_multi_agent_system",
]

# 向后兼容：保留旧名称
AgentGraph = ChatGraph  # type: ignore
create_agent_graph = create_chat_graph  # type: ignore
