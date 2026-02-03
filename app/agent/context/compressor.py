"""上下文压缩模块

提供智能的上下文压缩功能，保留重要信息。

使用示例:
```python
from app.agent.context.compressor import ContextCompressor, compress_context

# 使用压缩器
compressor = ContextCompressor(target_tokens=2000, model="gpt-4o")
compressed = await compressor.compress(messages)

# 使用便捷函数
compressed = await compress_context(messages, target_tokens=2000)
```
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agent.context.text_truncation import truncate_messages
from app.agent.context.token_counter import count_messages_tokens
from app.observability.logging import get_logger

logger = get_logger(__name__)


class ContextCompressor:
    """上下文压缩器

    压缩长对话历史，保留重要信息。
    """

    def __init__(
        self,
        target_tokens: int,
        model: str = "gpt-4o",
        preserve_system: bool = True,
        preserve_recent_n: int = 5,
    ):
        """初始化压缩器

        Args:
            target_tokens: 目标 Token 数
            model: 模型名称
            preserve_system: 是否保留系统消息
            preserve_recent_n: 保留最近 N 条消息
        """
        self.target_tokens = target_tokens
        self.model = model
        self.preserve_system = preserve_system
        self.preserve_recent_n = preserve_recent_n

    async def compress(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """压缩消息列表

        Args:
            messages: 消息列表

        Returns:
            压缩后的消息列表
        """
        if count_messages_tokens(messages, self.model) <= self.target_tokens:
            return messages

        # 分离系统消息
        system_message = None
        regular_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage) and system_message is None:
                system_message = msg
            else:
                regular_messages.append(msg)

        # 保留最近的消息
        recent_messages = regular_messages[-self.preserve_recent_n:]

        # 旧消息进行摘要压缩
        old_messages = regular_messages[:-self.preserve_recent_n]
        if old_messages:
            summary = await self._summarize_messages(old_messages)

            if summary:
                summary_msg = SystemMessage(content=f"[历史对话摘要]: {summary}")
                result = []

                if system_message and self.preserve_system:
                    result.append(system_message)

                result.append(summary_msg)
                result.extend(recent_messages)

                return result

        # 如果摘要失败，直接截断
        return truncate_messages(
            messages,
            self.target_tokens,
            self.model,
            self.preserve_system,
        )

    async def _summarize_messages(self, messages: list[BaseMessage]) -> str | None:
        """摘要消息列表

        Args:
            messages: 要摘要的消息

        Returns:
            摘要字符串
        """
        # 简单实现：提取关键信息
        # 生产环境可以使用 LLM 进行摘要

        summary_parts = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = str(msg.content)[:100]  # 限制长度
                summary_parts.append(f"用户: {content}...")
            elif isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    summary_parts.append(f"助手: 调用了 {len(msg.tool_calls)} 个工具")
                else:
                    content = str(msg.content)[:100]
                    summary_parts.append(f"助手: {content}...")

        return " | ".join(summary_parts)


async def compress_context(
    messages: list[BaseMessage],
    target_tokens: int,
    model: str = "gpt-4o",
) -> list[BaseMessage]:
    """压缩上下文

    Args:
        messages: 消息列表
        target_tokens: 目标 Token 数
        model: 模型名称

    Returns:
        压缩后的消息列表

    Examples:
        >>> compressed = await compress_context(messages, target_tokens=2000)
    """
    compressor = ContextCompressor(target_tokens, model)
    return await compressor.compress(messages)


__all__ = [
    "ContextCompressor",
    "compress_context",
]
