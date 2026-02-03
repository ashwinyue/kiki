"""Agent 状态定义

通用 Agent 状态（不使用 MessagesState）。
完全自定义的状态定义，适用于需要精细控制的场景。
"""

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """通用 Agent 状态（不使用 MessagesState）

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


__all__ = [
    "AgentState",
    "create_agent_state",
]
