"""图工具函数

提供图构建和执行过程中使用的工具函数。
"""

from typing import Any

from langchain_core.messages import HumanMessage

from app.observability.logging import get_logger

logger = get_logger(__name__)


def get_message_content(message: Any) -> str:
    """获取消息内容

    Args:
        message: 消息对象

    Returns:
        消息内容字符串
    """
    if isinstance(message, str):
        return message
    if hasattr(message, "content"):
        if isinstance(message.content, list):
            # 处理多模态内容
            text_parts = [
                part.get("text", "") for part in message.content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "".join(text_parts)
        return str(message.content)
    return str(message)


def is_user_message(message: Any) -> bool:
    """检查是否是用户消息

    Args:
        message: 消息对象

    Returns:
        是否是用户消息
    """
    if hasattr(message, "type"):
        return message.type == "human"
    if isinstance(message, HumanMessage):
        return True
    return False


def validate_state(state: dict[str, Any]) -> bool:
    """验证状态是否有效

    Args:
        state: 状态字典

    Returns:
        是否有效
    """
    if not isinstance(state, dict):
        return False

    if "messages" not in state:
        logger.warning("state_missing_messages")
        return False

    return True


def has_tool_calls(state: dict[str, Any]) -> bool:
    """检查状态中是否有工具调用

    Args:
        state: 状态字典

    Returns:
        是否有工具调用
    """
    messages = state.get("messages", [])
    if not messages:
        return False

    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return True

    return False


def should_continue(state: dict[str, Any]) -> bool:
    """判断是否应该继续执行

    检查迭代次数和错误状态。

    Args:
        state: 状态字典

    Returns:
        是否应该继续
    """
    # 检查错误
    if state.get("error"):
        return False

    # 检查迭代次数
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)

    return iteration_count < max_iterations


__all__ = [
    # 消息处理
    "get_message_content",
    "is_user_message",
    # 状态验证
    "validate_state",
    # 路由辅助
    "has_tool_calls",
    "should_continue",
]
