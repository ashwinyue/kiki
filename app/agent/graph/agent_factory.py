"""Agent 工厂 - 参考 DeerFlow 设计

使用 LangGraph 的 create_react_agent 创建标准化 Agent。
所有 Agent 都是 CompiledStateGraph，不需要额外的类包装。

设计原则：
- 简洁：统一函数创建所有 Agent
- 配置驱动：通过 AGENT_LLM_MAP 和 prompt_template 配置
- 标准化：使用 LangGraph 官方推荐的 create_react_agent

使用示例：
    ```python
    from app.agent.graph.agent_factory import create_agent

    # 创建 Planner
    planner = create_agent(
        agent_name="planner",
        agent_type="planner",
        tools=[],
        prompt_template="planner",
    )

    # 创建 Coder
    coder = create_agent(
        agent_name="coder",
        agent_type="coder",
        tools=[python_repl],
        prompt_template="coder",
    )
    ```
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from app.agent.config import AGENT_LLM_MAP, get_llm_by_type
from app.agent.prompts.template import render_prompt
from app.observability.logging import get_logger

logger = get_logger(__name__)


def create_agent(
    agent_name: str,
    agent_type: str,
    tools: list[BaseTool] | None = None,
    prompt_template: str | None = None,
    interrupt_before_tools: list[str] | None = None,
) -> Any:
    """统一 Agent 创建工厂（同步函数）

    参考 DeerFlow 设计，所有 Agent 都是 CompiledStateGraph。

    Args:
        agent_name: Agent 名称
        agent_type: Agent 类型（决定 LLM 类型）
        tools: 工具列表
        prompt_template: 提示词模板名称
        interrupt_before_tools: 需要拦截的工具列表（Phase 3）

    Returns:
        CompiledStateGraph: 编译后的 Agent 图

    Raises:
        ValueError: 如果 Agent 类型未映射
    """
    tools = tools or []

    # 获取 LLM 类型
    if agent_type not in AGENT_LLM_MAP:
        logger.warning(
            "agent_type_not_found",
            agent_type=agent_type,
            fallback="basic",
        )
        llm_type = "basic"
    else:
        llm_type = AGENT_LLM_MAP[agent_type]

    logger.debug(
        "creating_agent",
        agent_name=agent_name,
        agent_type=agent_type,
        llm_type=llm_type,
        tools_count=len(tools),
        template=prompt_template,
    )

    # 获取 LLM
    llm = get_llm_by_type(llm_type)

    # 工具拦截器（Phase 3 预留）
    processed_tools = tools
    if interrupt_before_tools:
        logger.warning(
            "interrupt_before_tools_not_implemented",
            agent_name=agent_name,
            tools=interrupt_before_tools,
        )

    # 构建 prompt 函数
    if prompt_template:
        # 使用 Jinja2 模板
        def prompt_fn(state: dict) -> str:
            locale = state.get("locale", "zh-CN")
            return render_prompt(
                prompt_template,
                locale=locale,
                agent_name=agent_name,
                tools=processed_tools,
                **state,
            )
    else:
        # 使用默认提示词
        def prompt_fn(state: dict) -> str:
            return f"你是 {agent_name}。"

    # 创建 ReAct Agent
    agent = create_react_agent(
        name=agent_name,
        model=llm,
        tools=processed_tools,
        prompt=prompt_fn,
    )

    logger.info(
        "agent_created",
        agent_name=agent_name,
        agent_type=agent_type,
        llm_type=llm_type,
    )

    return agent


__all__ = ["create_agent"]
