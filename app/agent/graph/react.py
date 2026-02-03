"""ReAct Agent 模块（DeerFlow 风格）

基于 LangGraph 的 create_react_agent 提供快速开发选项。

ReAct (Reasoning + Acting) 模式是一种经典的 Agent 模式，
LLM 通过推理决定采取什么行动，然后执行行动并观察结果。

DeerFlow 设计原则：
- 所有 Agent 都是 CompiledStateGraph
- 不需要额外的类包装
- 使用 LangGraph 官方的 create_react_agent
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent as langgraph_create_react_agent

from app.agent.config import get_llm_by_type
from app.agent.prompts.template import render_prompt
from app.observability.logging import get_logger

logger = get_logger(__name__)


def create_react_agent(
    agent_name: str = "react_agent",
    agent_type: str = "basic",
    tools: list[BaseTool] | None = None,
    prompt_template: str | None = None,
    **kwargs: Any,
) -> Any:
    """创建 ReAct Agent（返回 CompiledStateGraph）

    这是一个快速创建 Agent 的便捷函数，适合简单场景。

    Args:
        agent_name: Agent 名称
        agent_type: Agent 类型（决定 LLM 类型）
        tools: 工具列表
        prompt_template: 提示词模板名称
        **kwargs: 其他参数（传递给 langgraph_create_react_agent）

    Returns:
        CompiledStateGraph 实例

    Examples:
        ```python
        from langchain_core.tools import tool
        from app.agent.graph import create_react_agent

        @tool
        async def get_weather(location: str) -> str:
            \"\"\"获取指定位置的天气\"\"\"
            return f"{location} 今天晴天，25°C"

        agent = create_react_agent(
            agent_name="weather_agent",
            tools=[get_weather],
            prompt_template="chat",
        )

        # 调用 agent
        result = await agent.ainvoke(
            {"messages": [("user", "北京今天天气怎么样？")]},
            {"configurable": {"thread_id": "session-123"}},
        )
        ```
    """
    tools = tools or []

    logger.debug(
        "creating_react_agent",
        agent_name=agent_name,
        agent_type=agent_type,
        tools_count=len(tools),
    )

    # 获取 LLM
    llm = get_llm_by_type(agent_type)

    # 构建 prompt 函数
    if prompt_template:
        # 使用 Jinja2 模板
        def prompt_fn(state: dict) -> str:
            locale = state.get("locale", "zh-CN")
            return render_prompt(
                prompt_template,
                locale=locale,
                agent_name=agent_name,
                tools=tools,
                **state,
            )
    else:
        # 使用默认提示词
        def prompt_fn(state: dict) -> str:
            return f"你是 {agent_name}。"

    # 创建 ReAct Agent
    agent = langgraph_create_react_agent(
        name=agent_name,
        model=llm,
        tools=tools,
        prompt=prompt_fn,
        **kwargs,
    )

    logger.info("react_agent_created", agent_name=agent_name)

    return agent


__all__ = ["create_react_agent"]
