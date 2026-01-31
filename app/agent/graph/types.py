"""图状态定义

定义 LangGraph 工作流中使用的状态类型。

遵循 LangGraph 最佳实践：
- 使用 MessagesState 作为基类
- 使用 TypedDict 定义自定义状态
- 使用 Annotated 和 add_messages reducer 管理消息
"""

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 状态定义 ==============


class ChatState(MessagesState):
    """聊天状态（扩展 MessagesState）

    继承 LangGraph 的 MessagesState，自动包含 messages 字段
    并使用 add_messages reducer 管理消息历史。

    Attributes:
        messages: 消息列表（继承自 MessagesState，自动使用 add_messages reducer）
        user_id: 用户 ID
        session_id: 会话 ID
        tenant_id: 租户 ID
        iteration_count: 迭代计数（用于防止无限循环）
        max_iterations: 最大迭代次数
        error: 错误信息
    """

    # 用户和会话信息
    user_id: str | None
    session_id: str
    tenant_id: int | None

    # 迭代控制
    iteration_count: int
    max_iterations: int

    # 错误处理
    error: str | None


class AgentState(TypedDict):
    """通用 Agent 状态

    完全自定义的状态定义，适用于需要精细控制的场景。

    Attributes:
        messages: 消息列表（使用 add_messages reducer）
        query: 当前查询
        rewrite_query: 重写后的查询
        search_results: 搜索结果
        context_str: 构建的上下文
        iteration_count: 迭代计数
        max_iterations: 最大迭代次数
        error: 错误信息
    """

    # 消息历史（使用 add_messages reducer）
    messages: Annotated[list[BaseMessage], add_messages]

    # 查询相关
    query: str
    rewrite_query: str | None

    # 搜索和上下文
    search_results: list[Any]
    context_str: str

    # 控制字段
    iteration_count: int
    max_iterations: int

    # 错误处理
    error: str | None


class ReActState(TypedDict):
    """ReAct Agent 状态

    用于 ReAct 模式的 Agent，支持工具调用和推理。

    Attributes:
        messages: 消息列表（使用 add_messages reducer）
        tool_calls_to_execute: 待执行的工具调用
        iteration_count: 迭代计数
        max_iterations: 最大迭代次数
        error: 错误信息
    """

    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls_to_execute: list[dict[str, Any]]
    iteration_count: int
    max_iterations: int
    error: str | None


# ============== 状态工厂函数 ==============


def create_chat_state(
    messages: list[BaseMessage] | None = None,
    user_id: str | None = None,
    session_id: str = "",
    tenant_id: int | None = None,
) -> ChatState:
    """创建聊天状态

    Args:
        messages: 初始消息列表
        user_id: 用户 ID
        session_id: 会话 ID
        tenant_id: 租户 ID

    Returns:
        ChatState 实例
    """
    return ChatState(
        messages=messages or [],
        user_id=user_id,
        session_id=session_id,
        tenant_id=tenant_id,
        iteration_count=0,
        max_iterations=10,
        error=None,
    )


def create_agent_state(
    query: str = "",
    messages: list[BaseMessage] | None = None,
) -> AgentState:
    """创建 Agent 状态

    Args:
        query: 初始查询
        messages: 初始消息列表

    Returns:
        AgentState 实例
    """
    return AgentState(
        messages=messages or [],
        query=query,
        rewrite_query=None,
        search_results=[],
        context_str="",
        iteration_count=0,
        max_iterations=10,
        error=None,
    )


def create_react_state(
    messages: list[BaseMessage] | None = None,
) -> ReActState:
    """创建 ReAct 状态

    Args:
        messages: 初始消息列表

    Returns:
        ReActState 实例
    """
    return ReActState(
        messages=messages or [],
        tool_calls_to_execute=[],
        iteration_count=0,
        max_iterations=10,
        error=None,
    )


def create_state_from_input(
    input_data: str | dict[str, Any],
    session_id: str | None = None,
    user_id: str | None = None,
    tenant_id: int | None = None,
) -> ChatState:
    """从输入创建状态

    兼容旧版 API，返回 ChatState。

    Args:
        input_data: 输入数据（字符串或字典）
        session_id: 会话 ID
        user_id: 用户 ID
        tenant_id: 租户 ID

    Returns:
        ChatState 实例
    """
    if isinstance(input_data, str):
        query = input_data
    else:
        query = input_data.get("query", input_data.get("message", ""))

    from langchain_core.messages import HumanMessage

    messages = [HumanMessage(content=query)] if query else []

    return ChatState(
        messages=messages,
        user_id=user_id,
        session_id=session_id or "",
        tenant_id=tenant_id,
        iteration_count=0,
        max_iterations=10,
        error=None,
    )


# ============== 便捷函数 ==============


def increment_iteration(state: dict[str, Any]) -> dict[str, Any]:
    """增加迭代计数

    Args:
        state: 当前状态

    Returns:
        更新后的状态增量
    """
    current = state.get("iteration_count", 0)
    return {"iteration_count": current + 1}


def should_stop_iteration(state: dict[str, Any]) -> bool:
    """检查是否应该停止迭代

    Args:
        state: 当前状态

    Returns:
        是否应该停止
    """
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    return iteration_count >= max_iterations


def preserve_state_meta_fields(state: ChatState | AgentState) -> dict[str, Any]:
    """提取需要保留的元字段

    这些字段在状态转换时需要显式保留，避免恢复默认值。

    Args:
        state: 当前状态对象

    Returns:
        元字段字典
    """
    result = {
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 10),
    }

    # ChatState 特有字段
    if "user_id" in state:
        result["user_id"] = state.get("user_id")
    if "session_id" in state:
        result["session_id"] = state.get("session_id")
    if "tenant_id" in state:
        result["tenant_id"] = state.get("tenant_id")

    # AgentState 特有字段
    if "query" in state:
        result["query"] = state.get("query", "")
    if "rewrite_query" in state:
        result["rewrite_query"] = state.get("rewrite_query")
    if "search_results" in state:
        result["search_results"] = state.get("search_results", [])
    if "context_str" in state:
        result["context_str"] = state.get("context_str", "")

    return result


__all__ = [
    # 状态类
    "ChatState",
    "AgentState",
    "ReActState",
    # 工厂函数
    "create_chat_state",
    "create_agent_state",
    "create_react_state",
    "create_state_from_input",
    # 便捷函数
    "increment_iteration",
    "should_stop_iteration",
    "preserve_state_meta_fields",
    # 向后兼容
    "add_messages",
]
