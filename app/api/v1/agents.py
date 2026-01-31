"""多 Agent 协作 API

提供 Router、Supervisor、Swarm 等 Agent 模式的创建和交互接口。

此模块保留用于向后兼容，实际实现已拆分到 agents/ 子目录。
"""

# 从拆分后的模块导入，保持向后兼容
from app.api.v1.agents.__init__ import (
    AgentConfig,
    AgentSystemResponse,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    Message,
    RouterAgentRequest,
    SupervisorAgentRequest,
    SwarmAgentRequest,
    delete_agent_system,
    get_agent_system,
    list_agent_systems,
    router,
)

__all__ = [
    "router",
    "AgentConfig",
    "RouterAgentRequest",
    "SupervisorAgentRequest",
    "SwarmAgentRequest",
    "ChatRequest",
    "ChatResponse",
    "ChatHistoryResponse",
    "Message",
    "AgentSystemResponse",
    "get_agent_system",
    "list_agent_systems",
    "delete_agent_system",
]
