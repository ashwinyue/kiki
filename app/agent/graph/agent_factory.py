"""统一 Agent 工厂

参考 DeerFlow 的 Agent 创建模式，使用 LangGraph 的 create_react_agent。
支持：
- 分层 LLM 配置
- 工具拦截器
- Jinja2 提示词模板（Phase 2 预留）

核心优势：
1. **标准化**: 使用 LangGraph 官方推荐的 create_react_agent
2. **分层配置**: 不同 Agent 类型使用不同 LLM
3. **灵活扩展**: 易于添加工具、拦截器和模板

使用示例：
    ```python
    from app.agent.graph.agent_factory import create_agent
    from app.agent.tools import alist_tools

    # 创建 Planner Agent
    planner = await create_agent(
        agent_name="planner",
        agent_type="planner",
        tools=[],
        prompt_template="planner",
    )

    # 创建 Coder Agent（带工具）
    tools = await alist_tools()
    coder = await create_agent(
        agent_name="coder",
        agent_type="coder",
        tools=tools,
        prompt_template="coder",
    )
    ```
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from app.agent.config import get_llm_for_agent
from app.observability.logging import get_logger

logger = get_logger(__name__)


async def create_agent(
    agent_name: str,
    agent_type: str,
    tools: list[BaseTool] | None = None,
    prompt_template: str | None = None,
    system_prompt: str | None = None,
    interrupt_before_tools: list[str] | None = None,
    state_modifier: str | Literal["messages", "last_message"] = "messages",
    **llm_kwargs,
) -> Any:
    """统一 Agent 创建工厂

    使用 LangGraph 的 create_react_agent 创建标准化 Agent。
    支持分层 LLM 配置和工具拦截器。

    Args:
        agent_name: Agent 名称（用于日志和追踪）
        agent_type: Agent 类型（决定 LLM 类型）
            - 支持: planner, coder, researcher, analyst, reporter, supervisor, router, chat, worker
            - 完整列表见 app/agent/config/llm_config.py:AGENT_LLM_MAP
        tools: 工具列表（可选）
        prompt_template: 提示词模板名称（Phase 2，预留）
        system_prompt: 系统提示词（当前使用，Phase 2 后被模板替代）
        interrupt_before_tools: 需要拦截的工具列表（Phase 3，预留）
        state_modifier: 状态修改器类型
            - "messages": 完整消息历史
            - "last_message": 仅最后一条消息
        **llm_kwargs: LLM 参数覆盖（temperature, max_tokens 等）

    Returns:
        CompiledStateGraph: 编译后的 Agent 图

    Raises:
        ValueError: 如果 Agent 类型未映射或 LLM 创建失败

    Examples:
        ```python
        # 创建 Planner Agent（无需工具）
        planner = await create_agent(
            agent_name="planner",
            agent_type="planner",
            system_prompt="你是一个任务规划专家。",
        )

        # 创建 Coder Agent（带工具）
        tools = await alist_tools()
        coder = await create_agent(
            agent_name="coder",
            agent_type="coder",
            tools=tools,
            system_prompt="你是一个代码专家。",
            temperature=0.1,  # 覆盖默认温度
        )
        ```
    """
    # 1. 获取 LLM（根据 agent_type）
    try:
        llm = get_llm_for_agent(agent_type, **llm_kwargs)
    except ValueError as e:
        logger.error(
            "agent_creation_failed",
            agent_name=agent_name,
            agent_type=agent_type,
            error=str(e),
        )
        raise

    # 2. 处理工具（拦截器预留）
    processed_tools = tools or []
    if interrupt_before_tools:
        # TODO: Phase 3 - 集成工具拦截器
        logger.warning(
            "interrupt_before_tools_not_implemented",
            agent_name=agent_name,
            tools=interrupt_before_tools,
        )

    # 3. 构建提示词
    if system_prompt:
        # 当前使用硬编码提示词（Phase 2 后迁移到 Jinja2）
        prompt = _build_prompt(system_prompt, state_modifier)
    elif prompt_template:
        # TODO: Phase 2 - Jinja2 模板渲染
        logger.warning(
            "prompt_template_not_implemented",
            agent_name=agent_name,
            template=prompt_template,
            fallback="using_system_prompt",
        )
        prompt = _build_prompt(f"你是 {agent_name}。", state_modifier)
    else:
        prompt = _build_prompt(f"你是 {agent_name}。", state_modifier)

    # 4. 创建 ReAct Agent
    try:
        agent_graph = create_react_agent(
            model=llm,
            tools=processed_tools,
            prompt=prompt,
            state_modifier=state_modifier,
        )

        logger.info(
            "react_agent_created",
            agent_name=agent_name,
            agent_type=agent_type,
            llm_model=llm.model if hasattr(llm, "model") else "unknown",
            tools_count=len(processed_tools),
        )

        return agent_graph

    except Exception as e:
        logger.error(
            "react_agent_creation_failed",
            agent_name=agent_name,
            agent_type=agent_type,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise


def _build_prompt(
    system_prompt: str,
    state_modifier: Literal["messages", "last_message"] = "messages",
) -> Any:
    """构建 Agent 提示词

    Args:
        system_prompt: 系统提示词
        state_modifier: 状态修改器类型

    Returns:
        提示词函数（兼容 create_react_agent）
    """
    # LangGraph 的 create_react_agent 接受 callable 或 str
    # 这里返回一个函数，符合 LangGraph 的 prompt 参数要求
    def prompt_fn(state: dict[str, Any]) -> list[dict[str, str]]:
        """提示词函数

        根据状态生成消息列表。
        """
        messages = []

        # 添加系统提示词
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # 添加历史消息
        if state_modifier == "messages":
            history = state.get("messages", [])
            for msg in history:
                if hasattr(msg, "to_dict"):
                    messages.append(msg.to_dict())
                elif isinstance(msg, dict):
                    messages.append(msg)
                else:
                    messages.append({
                        "role": "user",
                        "content": str(msg),
                    })
        elif state_modifier == "last_message":
            history = state.get("messages", [])
            if history:
                last_msg = history[-1]
                if hasattr(last_msg, "to_dict"):
                    messages.append(last_msg.to_dict())
                elif isinstance(last_msg, dict):
                    messages.append(last_msg)
                else:
                    messages.append({
                        "role": "user",
                        "content": str(last_msg),
                    })

        return messages

    return prompt_fn


# ============== 批量创建 Agents ==============

async def create_agents(
    agent_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """批量创建 Agents

    Args:
        agent_configs: Agent 配置字典
            ```python
            {
                "planner": {
                    "agent_type": "planner",
                    "system_prompt": "你是任务规划专家",
                    "tools": [],
                },
                "coder": {
                    "agent_type": "coder",
                    "system_prompt": "你是代码专家",
                    "tools": [python_repl],
                },
            }
            ```

    Returns:
        {agent_name: agent_graph} 字典

    Examples:
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
        }
        agents = await create_agents(configs)

        # 使用 planner
        planner = agents["planner"]
        result = await planner.ainvoke({...})
        ```
    """
    agents: dict[str, Any] = {}

    for agent_name, config in agent_configs.items():
        try:
            agent = await create_agent(
                agent_name=agent_name,
                **config,
            )
            agents[agent_name] = agent
        except Exception as e:
            logger.error(
                "batch_agent_creation_failed",
                agent_name=agent_name,
                error=str(e),
            )
            # 继续创建其他 agents
            continue

    logger.info(
        "batch_agents_created",
        total_requested=len(agent_configs),
        total_created=len(agents),
        failed=len(agent_configs) - len(agents),
    )

    return agents


# ============== 专门化角色创建函数（DeerFlow 风格）==========

async def create_planner_agent(
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> Any:
    """创建 Planner Agent

    Planner 负责任务分解和计划生成。

    Args:
        tools: 工具列表（通常 Planner 不需要工具）
        system_prompt: 自定义提示词
        **kwargs: 其他参数

    Returns:
        编译后的 Planner Agent
    """
    default_prompt = """你是一个任务规划专家。

你的职责：
1. 分析用户的目标
2. 将复杂任务分解为可执行的步骤
3. 创建结构化的执行计划

输出格式：
- 使用清晰的步骤编号
- 每个步骤包含具体行动
- 考虑依赖关系和执行顺序
"""
    return await create_agent(
        agent_name="planner",
        agent_type="planner",
        tools=tools or [],
        system_prompt=system_prompt or default_prompt,
        **kwargs,
    )


async def create_researcher_agent(
    tools: list[BaseTool],
    system_prompt: str | None = None,
    **kwargs,
) -> Any:
    """创建 Researcher Agent

    Researcher 负责信息检索和网络搜索。

    Args:
        tools: 工具列表（应包含搜索工具）
        system_prompt: 自定义提示词
        **kwargs: 其他参数

    Returns:
        编译后的 Researcher Agent
    """
    default_prompt = """你是一个信息检索专家。

你的职责：
1. 使用搜索工具查找相关信息
2. 验证信息来源的可靠性
3. 整理和总结搜索结果

搜索策略：
- 使用多个关键词
- 交叉验证信息
- 关注最新信息
"""
    return await create_agent(
        agent_name="researcher",
        agent_type="researcher",
        tools=tools,
        system_prompt=system_prompt or default_prompt,
        **kwargs,
    )


async def create_analyst_agent(
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> Any:
    """创建 Analyst Agent

    Analyst 负责数据分析和推理。

    Args:
        tools: 工具列表（可选）
        system_prompt: 自定义提示词
        **kwargs: 其他参数

    Returns:
        编译后的 Analyst Agent
    """
    default_prompt = """你是一个数据分析专家。

你的职责：
1. 分析提供的数据和信息
2. 识别模式和趋势
3. 得出合理的结论

分析方法：
- 结构化思考
- 多角度分析
- 验证假设
"""
    return await create_agent(
        agent_name="analyst",
        agent_type="analyst",
        tools=tools or [],
        system_prompt=system_prompt or default_prompt,
        **kwargs,
    )


async def create_coder_agent(
    tools: list[BaseTool],
    system_prompt: str | None = None,
    **kwargs,
) -> Any:
    """创建 Coder Agent

    Coder 负责代码生成和执行。

    Args:
        tools: 工具列表（应包含 Python REPL 或代码执行工具）
        system_prompt: 自定义提示词
        **kwargs: 其他参数

    Returns:
        编译后的 Coder Agent
    """
    default_prompt = """你是一个代码专家。

你的职责：
1. 编写高质量的代码
2. 解释代码逻辑
3. 调试和优化代码

编程规范：
- 代码清晰易读
- 添加必要注释
- 遵循最佳实践
- 考虑边界情况
"""
    return await create_agent(
        agent_name="coder",
        agent_type="coder",
        tools=tools,
        system_prompt=system_prompt or default_prompt,
        **kwargs,
    )


async def create_reporter_agent(
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> Any:
    """创建 Reporter Agent

    Reporter 负责聚合结果和生成报告。

    Args:
        tools: 工具列表（可选）
        system_prompt: 自定义提示词
        **kwargs: 其他参数

    Returns:
        编译后的 Reporter Agent
    """
    default_prompt = """你是一个报告生成专家。

你的职责：
1. 聚合各个 Agent 的输出
2. 生成结构化的报告
3. 突出关键发现和结论

报告格式：
- 清晰的章节结构
- 突出重点内容
- 提供可操作的建议
"""
    return await create_agent(
        agent_name="reporter",
        agent_type="reporter",
        tools=tools or [],
        system_prompt=system_prompt or default_prompt,
        **kwargs,
    )


__all__ = [
    # 核心
    "create_agent",
    "create_agents",
    # 专门化角色
    "create_planner_agent",
    "create_researcher_agent",
    "create_analyst_agent",
    "create_coder_agent",
    "create_reporter_agent",
]
