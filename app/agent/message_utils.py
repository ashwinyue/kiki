"""消息工具函数

提供消息处理和内容提取的工具函数。
"""

from langchain_core.messages import BaseMessage


def extract_ai_content(messages: list[BaseMessage]) -> str:
    """从消息列表中提取 AI 响应内容

    Args:
        messages: 消息列表

    Returns:
        AI 响应内容
    """
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content
        elif hasattr(msg, "content"):
            return str(msg.content)
    return ""


def get_last_user_message(messages: list[BaseMessage]) -> str:
    """获取最后一条用户消息

    Args:
        messages: 消息列表

    Returns:
        用户消息内容
    """
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""


def format_messages_to_dict(
    messages: list[BaseMessage],
    role_map: dict[str, str] | None = None,
) -> list[dict]:
    """将消息转换为字典列表

    Args:
        messages: 消息列表
        role_map: 角色映射，默认 {"human": "user", "ai": "assistant"}

    Returns:
        消息字典列表
    """
    if role_map is None:
        role_map = {"human": "user", "ai": "assistant", "system": "system"}

    result = []
    for msg in messages:
        if hasattr(msg, "type") and msg.type in role_map:
            result.append({
                "role": role_map.get(msg.type, msg.type),
                "content": str(msg.content),
            })
    return result
