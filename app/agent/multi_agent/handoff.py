"""Handoff (Swarm) Agent 模式

Agent 之间可以主动切换控制权。

示意图:
┌───────┐ handoff   ┌───────┐
│ Alice │──────────→│  Bob  │
└───────┘           └───────┘
    ↑                   │
    └───────────────────┘
         handoff
"""

from typing import Any

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import RunnableConfig

from app.agent.state import AgentState
from app.llm import LLMService
from app.observability.logging import get_logger

logger = get_logger(__name__)


def create_handoff_tool(
    agent_name: str,
    description: str | None = None,
) -> Any:
    """创建 Agent 切换工具

    Args:
        agent_name: 目标 Agent 名称
        description: 工具描述

    Returns:
        LangChain 工具实例

    Examples:
        ```python
        transfer_to_bob = create_handoff_tool(
            agent_name="Bob",
            description="Transfer to Bob for technical support"
        )
        ```
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    actual_description = description or f"Transfer to {agent_name}"

    class HandoffInput(BaseModel):
        """Handoff 输入"""

        reason: str = Field(description="切换原因")

    async def handoff_wrapper(reason: str) -> str:
        """包装函数供 LLM 调用"""
        return f"Transferring to {agent_name}. Reason: {reason}"

    tool = StructuredTool.from_function(
        func=handoff_wrapper,
        name=f"transfer_to_{agent_name.lower()}",
        description=actual_description,
        args_schema=HandoffInput,
    )

    # 保存元数据
    tool._handoff_target = agent_name

    return tool


class HandoffAgent:
    """支持动态切换的 Agent

    Agent 可以主动将控制权切换给其他 Agent。

    使用标准 add_conditional_edges 模式实现切换。
    """

    # 特殊标记：表示切换到特定 Agent
    HANDOFF_PREFIX = "handoff_to_"

    def __init__(
        self,
        name: str,
        llm_service: LLMService,
        tools: list = None,
        handoff_targets: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """初始化可切换 Agent

        Args:
            name: Agent 名称
            llm_service: LLM 服务
            tools: 工具列表
            handoff_targets: 可以切换到的目标 Agent 名称
            system_prompt: 系统提示词
        """
        self.name = name
        self._llm_service = llm_service
        self._tools = tools or []
        self._handoff_targets = handoff_targets or []
        self._system_prompt = system_prompt or f"You are {name}, a helpful assistant."
        self._handoff_tools: list = []

        # 创建切换工具
        for target in self._handoff_targets:
            handoff_tool = create_handoff_tool(target)
            self._handoff_tools.append(handoff_tool)

    async def _node(
        self,
        state: AgentState,
        config: RunnableConfig,
    ) -> dict:
        """Agent 节点

        检测切换请求并将切换目标存储在状态中。

        Returns:
            状态更新
        """
        # 准备消息
        messages = list(state["messages"])

        # 添加系统提示
        if not any(m.type == "system" for m in messages):
            messages.insert(0, SystemMessage(content=self._system_prompt))

        # 绑定所有工具（包括切换工具）
        all_tools = self._tools + self._handoff_tools
        llm_with_tools = self._llm_service.get_llm_with_tools(all_tools)

        # 调用 LLM
        if llm_with_tools:
            response = await llm_with_tools.ainvoke(messages)
        else:
            response = await self._llm_service.call(messages)

        # 检查是否是切换请求
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                if "transfer_to_" in tool_name:
                    # 执行切换
                    target = tool_name.replace("transfer_to_", "").capitalize()
                    logger.info("agent_initiated_handoff", from_agent=self.name, to=target)
                    return {"_handoff_target": target, "messages": [response]}

        # 正常响应，继续到 END
        return {"messages": [response], "_handoff_target": None}

    def get_handoff_targets(self) -> list[str]:
        """获取可切换的目标 Agent 列表"""
        return self._handoff_targets.copy()


def create_swarm(
    agents: list[HandoffAgent],
    default_agent: str,
) -> StateGraph:
    """创建 Agent Swarm

    使用条件边实现 Agent 之间的动态切换。

    Args:
        agents: HandoffAgent 列表
        default_agent: 默认激活的 Agent

    Returns:
        编译后的 StateGraph
    """
    builder = StateGraph(AgentState)

    # 收集所有 Agent 名称
    agent_names = [agent.name for agent in agents]

    def handoff_edge(state: AgentState) -> str:
        """切换边函数 - 决定下一个 Agent"""
        target = state.get("_handoff_target", "")
        if target and target in agent_names:
            return target
        return END

    # 添加所有 Agent 节点
    for agent in agents:
        builder.add_node(agent.name, agent._node)

    # 设置入口
    builder.set_entry_point(default_agent)

    # 为每个 Agent 添加条件边：可以切换到其他 Agent 或结束
    route_mapping = {name: name for name in agent_names}
    route_mapping[None] = END  # 无切换目标时结束
    for agent_name in agent_names:
        builder.add_conditional_edges(agent_name, handoff_edge, route_mapping)

    return builder.compile()
