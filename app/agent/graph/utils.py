"""图工具函数

提供图构建和执行过程中使用的工具函数。
"""

from typing import Any

from langchain_core.messages import HumanMessage

from app.observability.logging import get_logger

logger = get_logger(__name__)


# ============== 消息处理 ==============


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


def format_messages_to_dict(messages: list[Any]) -> list[dict[str, str]]:
    """将消息列表转换为字典列表

    Args:
        messages: 消息列表

    Returns:
        字典列表 [{"role": ..., "content": ...}, ...]
    """
    result = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = get_message_content(msg)

        # 转换角色名称
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        role = role_map.get(role, role)

        result.append({"role": role, "content": content})
    return result


def extract_ai_content(messages: list[Any]) -> str:
    """提取最后一条 AI 消息的内容

    Args:
        messages: 消息列表

    Returns:
        AI 消息内容，如果没有则返回空字符串
    """
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            return get_message_content(msg)
    return ""


# ============== 状态验证 ==============


def validate_state(state: dict[str, Any]) -> bool:
    """验证状态是否有效

    Args:
        state: 状态字典

    Returns:
        是否有效
    """
    if not isinstance(state, dict):
        return False

    # 检查是否有 messages 字段
    if "messages" not in state:
        logger.warning("state_missing_messages")
        return False

    return True


# ============== 路由辅助 ==============


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
    "format_messages_to_dict",
    "extract_ai_content",
    # 状态验证
    "validate_state",
    # 路由辅助
    "has_tool_calls",
    "should_continue",
]
