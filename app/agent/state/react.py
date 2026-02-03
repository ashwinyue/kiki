"""ReAct Agent 状态定义

用于 ReAct 模式的 Agent，支持工具调用和推理。
"""

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


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


__all__ = [
    "ReActState",
    "create_react_state",
]
