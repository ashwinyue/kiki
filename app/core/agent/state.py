"""Agent 状态定义

使用 LangGraph 的 add_messages reducer 实现消息历史的自动追加。
"""

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Agent 状态

    使用 LangGraph 的 add_messages reducer 实现消息历史的自动管理。

    Attributes:
        messages: 消息历史（使用 add_messages reducer）
        user_id: 用户 ID（用于多租户和长期记忆）
        session_id: 会话 ID（用于状态恢复）
    """

    # 消息历史（使用 add_messages reducer 自动追加）
    messages: Annotated[list[BaseMessage], add_messages]

    # 元数据
    user_id: str | None
    session_id: str | None


def create_initial_state(
    messages: list[BaseMessage] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """创建初始状态

    Args:
        messages: 初始消息列表
        user_id: 用户 ID
        session_id: 会话 ID

    Returns:
        初始状态字典
    """
    return {
        "messages": messages or [],
        "user_id": user_id,
        "session_id": session_id,
    }


def create_state_from_input(
    input_text: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """从用户输入创建状态

    Args:
        input_text: 用户输入文本
        user_id: 用户 ID
        session_id: 会话 ID

    Returns:
        初始状态字典
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=input_text)],
        "user_id": user_id,
        "session_id": session_id,
    }
