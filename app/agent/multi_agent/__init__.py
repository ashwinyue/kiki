"""多 Agent 系统

支持多种多 Agent 模式：
1. Router Agent - 路由到不同的子 Agent
2. Supervisor Agent - 监督多个 Worker Agent
3. Handoff - Agent 之间动态切换

使用 LangChain 的 with_structured_output 实现类型安全的路由决策。
使用 add_conditional_edges 实现标准的条件边路由模式。
"""

from typing import Any

from langgraph.graph import StateGraph

from app.agent.multi_agent.handoff import HandoffAgent, create_handoff_tool, create_swarm
from app.agent.multi_agent.router import RouterAgent
from app.agent.multi_agent.supervisor import SupervisorAgent
from app.llm import LLMService

__all__ = [
    "RouterAgent",
    "SupervisorAgent",
    "HandoffAgent",
    "create_handoff_tool",
    "create_swarm",
    "create_multi_agent_system",
]


def create_multi_agent_system(
    mode: str = "router",
    llm_service: LLMService | None = None,
    **kwargs,
) -> StateGraph:
    """创建多 Agent 系统的便捷函数

    Args:
        mode: 多 Agent 模式 (router/supervisor/swarm)
        llm_service: LLM 服务
        **kwargs: 模式特定参数

    Returns:
        编译后的 StateGraph
    """
    if mode == "router":
        from app.llm import get_llm_service

        if llm_service is None:
            llm_service = get_llm_service()

        agents = kwargs.get("agents", {})
        return RouterAgent(llm_service, agents).compile()

    elif mode == "supervisor":
        from app.llm import get_llm_service

        if llm_service is None:
            llm_service = get_llm_service()

        workers = kwargs.get("workers", {})
        return SupervisorAgent(llm_service, workers).compile()

    elif mode == "swarm":
        agents = kwargs.get("agents", [])
        default_agent = kwargs.get("default_agent", agents[0].name if agents else "Agent")
        return create_swarm(agents, default_agent)

    else:
        raise ValueError(f"Unknown multi-agent mode: {mode}")
