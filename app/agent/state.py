"""Agent 状态定义

使用 LangGraph 的 add_messages reducer 实现消息历史的自动追加。
参考 DeerFlow 的 State 设计，增强企业级功能。
"""

from dataclasses import field
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


# 自定义 reducer：确保迭代计数正确累加
def _add_iteration(left: int | None, right: int | None) -> int:
    """迭代计数累加器

    Args:
        left: 当前计数值
        right: 要增加的值

    Returns:
        累加后的计数值
    """
    left_count = left or 0
    right_count = right or 0
    return left_count + right_count


def _merge_lists(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """列表合并 reducer（去重）

    Args:
        left: 当前列表
        right: 要添加的列表

    Returns:
        合并后的列表（去重）
    """
    left_list = left or []
    right_list = right or []
    merged = left_list + right_list
    # 简单去重（适用于可哈希元素）
    try:
        return list(dict.fromkeys(merged))
    except TypeError:
        # 如果有不可哈希的元素，直接返回合并结果
        return merged


def _trim_messages(
    left: list[BaseMessage] | None,
    right: list[BaseMessage] | None,
) -> list[BaseMessage]:
    """消息滑动窗口 Reducer

    自动修剪超过窗口大小的消息历史，保留最近的消息。
    始终保留第一条系统消息（如果有）。

    Args:
        left: 当前消息列表
        right: 新增的消息列表

    Returns:
        修剪后的消息列表
    """
    from app.core.config import get_settings

    settings = get_settings()
    max_messages = settings.agent_max_messages  # 从配置读取最大消息数

    # 合并消息
    left_messages = left or []
    right_messages = right or []
    merged = left_messages + right_messages

    # 如果消息数量未超过限制，直接返回
    if len(merged) <= max_messages:
        return merged

    # 检查是否有系统消息需要保留
    system_message = None
    other_messages = []
    for msg in merged:
        if msg.type == "system" and system_message is None:
            system_message = msg
        else:
            other_messages.append(msg)

    # 保留最近的消息（不包括系统消息）
    # 计算可保留的非系统消息数量
    available_slots = max_messages - (1 if system_message else 0)
    trimmed_other = (
        other_messages[-available_slots:]
        if len(other_messages) > available_slots
        else other_messages
    )

    # 组合结果
    if system_message:
        return [system_message] + trimmed_other
    return trimmed_other


class AgentState(TypedDict, total=False):
    """Agent 状态

    使用 LangGraph 的 add_messages reducer 实现消息历史的自动管理。
    使用滑动窗口 reducer 自动修剪超过限制的消息。
    参考 DeerFlow 的 State 设计，增强企业级功能。

    Attributes:
        messages: 消息历史（使用滑动窗口 reducer 自动修剪）
        user_id: 用户 ID（用于多租户和长期记忆）
        session_id: 会话 ID（用于状态恢复）
        agent_id: 当前 Agent ID
        locale: 语言环境（默认 "zh-CN"）

        # 迭代控制（防止无限循环）
        iteration_count: 当前迭代次数
        max_iterations: 最大允许迭代次数（默认 50）

        # 任务状态
        current_task: 当前任务描述
        observations: 观察记录列表
        final_output: 最终输出

        # 工作流控制
        goto: 下一个节点（用于工作流路由）
        next_node: 下一个节点（别名）
        _next_agent: 内部路由决策字段（用于 Router Agent）
        _next_worker: 内部监督决策字段（用于 Supervisor Agent）
        _handoff_target: 内部切换目标字段（用于 Handoff Agent）

        # 工具控制
        interrupt_before_tools: 需要中断执行的工具名称列表
        tool_execution_history: 工具执行历史

        # 澄清机制（多轮对话）
        enable_clarification: 是否启用澄清功能
        clarification_rounds: 澄清轮次
        clarification_history: 澄清历史
        max_clarification_rounds: 最大澄清轮次（默认 3）
        is_clarification_complete: 澄清是否完成

        # 窗口记忆（Token 限制）
        llm_input_messages: 修剪后用于 LLM 输入的消息
        window_max_tokens: 窗口最大 Token 数
        window_strategy: 窗口修剪策略（last/first）

        # 扩展元数据
        metadata: 扩展元数据字典
    """

    # ========== 消息历史 ==========
    messages: Annotated[list[BaseMessage], _trim_messages]

    # ========== 用户上下文 ==========
    user_id: str | None
    session_id: str | None
    agent_id: str | None
    locale: str

    # ========== 迭代控制 ==========
    iteration_count: Annotated[int, _add_iteration]
    max_iterations: int

    # ========== 任务状态 ==========
    current_task: dict[str, Any] | None
    observations: Annotated[list[str], _merge_lists]
    final_output: str | None

    # ========== 工作流控制 ==========
    goto: str
    next_node: str | None
    _next_agent: str | None
    _next_worker: str | None
    _handoff_target: str | None

    # ========== 工具控制 ==========
    interrupt_before_tools: list[str]
    tool_execution_history: list[dict[str, Any]]

    # ========== 澄清机制 ==========
    enable_clarification: bool
    clarification_rounds: int
    clarification_history: list[str]
    max_clarification_rounds: int
    is_clarification_complete: bool

    # ========== 窗口记忆 ==========
    llm_input_messages: list[BaseMessage] | None
    window_max_tokens: int
    window_strategy: str

    # ========== 扩展元数据 ==========
    metadata: dict[str, Any]


def create_initial_state(
    messages: list[BaseMessage] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    max_iterations: int | None = None,
    locale: str = "zh-CN",
    enable_clarification: bool = False,
    interrupt_before_tools: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """创建初始状态

    Args:
        messages: 初始消息列表
        user_id: 用户 ID
        session_id: 会话 ID
        agent_id: Agent ID
        max_iterations: 最大迭代次数（防止无限循环）
        locale: 语言环境
        enable_clarification: 是否启用澄清功能
        interrupt_before_tools: 需要中断的工具列表
        metadata: 扩展元数据

    Returns:
        初始状态字典
    """
    from app.core.config import get_settings

    settings = get_settings()
    resolved_max_iterations = max_iterations or settings.agent_max_iterations

    return {
        "messages": messages or [],
        "user_id": user_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "locale": locale,
        "iteration_count": 0,
        "max_iterations": resolved_max_iterations,
        "current_task": None,
        "observations": [],
        "final_output": None,
        "goto": "",
        "next_node": None,
        "_next_agent": None,
        "_next_worker": None,
        "_handoff_target": None,
        "interrupt_before_tools": interrupt_before_tools or [],
        "tool_execution_history": [],
        "enable_clarification": enable_clarification,
        "clarification_rounds": 0,
        "clarification_history": [],
        "max_clarification_rounds": resolved_max_iterations // 10,
        "is_clarification_complete": False,
        "llm_input_messages": None,
        "window_max_tokens": settings.context_max_tokens,
        "window_strategy": "last",
        "metadata": metadata or {},
    }


def create_state_from_input(
    input_text: str,
    user_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    max_iterations: int | None = None,
    locale: str = "zh-CN",
    enable_clarification: bool = False,
) -> dict[str, Any]:
    """从用户输入创建状态

    Args:
        input_text: 用户输入文本
        user_id: 用户 ID
        session_id: 会话 ID
        agent_id: Agent ID
        max_iterations: 最大迭代次数（防止无限循环）
        locale: 语言环境
        enable_clarification: 是否启用澄清功能

    Returns:
        初始状态字典
    """
    from langchain_core.messages import HumanMessage

    return {
        **create_initial_state(
            messages=[HumanMessage(content=input_text)],
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            max_iterations=max_iterations,
            locale=locale,
            enable_clarification=enable_clarification,
        ),
    }


def get_default_state() -> dict[str, Any]:
    """获取默认状态（空状态）

    用于初始化或重置状态。

    Returns:
        默认状态字典
    """
    return create_initial_state()
