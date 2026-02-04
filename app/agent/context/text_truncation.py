"""文本截断模块

提供消息列表和文本的智能截断功能。
"""

from langchain_core.messages import BaseMessage, SystemMessage

from app.agent.context.token_counter import count_messages_tokens, count_tokens
from app.observability.logging import get_logger

logger = get_logger(__name__)


def truncate_messages(
    messages: list[BaseMessage],
    max_tokens: int,
    model: str = "gpt-4o",
    keep_system_first: bool = True,
) -> list[BaseMessage]:
    """截断消息列表以适应 Token 限制

    策略：
    1. 始终保留第一条系统消息（如果有）
    2. 保留最近的消息
    3. 移除最早的非系统消息

    Args:
        messages: 消息列表
        max_tokens: 最大 Token 数
        model: 模型名称
        keep_system_first: 是否保留第一条系统消息

    Returns:
        截断后的消息列表

    Examples:
        >>> messages = [SystemMessage("系统"), HumanMessage("问题"), AIMessage("回答")]
        >>> truncate_messages(messages, max_tokens=100)
        [SystemMessage("系统"), HumanMessage("问题"), AIMessage("回答")]
    """
    if not messages:
        return []

    # 分离系统消息和普通消息
    system_message = None
    regular_messages = []

    for msg in messages:
        if isinstance(msg, SystemMessage) and system_message is None:
            system_message = msg
        else:
            regular_messages.append(msg)

    # 估算系统消息的 Token 数
    system_tokens = 0
    if system_message:
        system_tokens = count_messages_tokens([system_message], model)

    # 从后向前添加消息，直到达到限制
    result_regular = []
    current_tokens = system_tokens

    for msg in reversed(regular_messages):
        msg_tokens = count_messages_tokens([msg], model)

        if current_tokens + msg_tokens > max_tokens:
            break

        result_regular.insert(0, msg)
        current_tokens += msg_tokens

    # 组合结果
    result = []
    if system_message and keep_system_first:
        result.append(system_message)

    result.extend(result_regular)

    logger.info(
        "messages_truncated",
        original_count=len(messages),
        result_count=len(result),
        original_tokens=count_messages_tokens(messages, model),
        result_tokens=count_messages_tokens(result, model),
        max_tokens=max_tokens,
    )

    return result


def truncate_text(
    text: str,
    max_tokens: int,
    model: str = "gpt-4o",
    add_ellipsis: bool = True,
) -> str:
    """截断文本以适应 Token 限制

    Args:
        text: 输入文本
        max_tokens: 最大 Token 数
        model: 模型名称
        add_ellipsis: 是否添加省略号

    Returns:
        截断后的文本

    Examples:
        >>> truncate_text("这是一段很长的文本..." * 100, max_tokens=100)
        '这是一段很长的文本......'
    """
    if count_tokens(text, model) <= max_tokens:
        return text

    # 二分查找最大长度
    left, right = 0, len(text)

    while left < right:
        mid = (left + right + 1) // 2
        truncated = text[:mid]

        if count_tokens(truncated, model) <= max_tokens:
            left = mid
        else:
            right = mid - 1

    result = text[:left]

    if add_ellipsis and len(result) < len(text):
        # 确保省略号不会超过限制
        ellipsis = "..."
        ellipsis_tokens = count_tokens(ellipsis, model)

        if count_tokens(result, model) + ellipsis_tokens <= max_tokens:
            result += ellipsis
        else:
            # 进一步缩短以容纳省略号
            result = result[: max(0, left - 10)]
            result += ellipsis

    logger.debug(
        "text_truncated",
        original_length=len(text),
        result_length=len(result),
        max_tokens=max_tokens,
    )

    return result


__all__ = [
    "truncate_messages",
    "truncate_text",
]
