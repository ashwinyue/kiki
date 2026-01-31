"""多 Agent 系统

支持多种多 Agent 模式：
1. Router Agent - 路由到不同的子 Agent
2. Supervisor Agent - 监督多个 Worker Agent
3. Handoff - Agent 之间动态切换

此模块保留用于向后兼容，实际实现已拆分到 multi_agent/ 子模块。
"""

# 从拆分后的模块导入，保持向后兼容
from app.agent.multi_agent.__init__ import (
    HandoffAgent,
    RouterAgent,
    SupervisorAgent,
    create_handoff_tool,
    create_multi_agent_system,
    create_swarm,
)

__all__ = [
    "RouterAgent",
    "SupervisorAgent",
    "HandoffAgent",
    "create_handoff_tool",
    "create_swarm",
    "create_multi_agent_system",
]
