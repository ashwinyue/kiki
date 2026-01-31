"""长文本处理模块

提供 Token 计算和智能截断功能，处理长对话历史和长文档。
支持中英文 Token 计算，自动压缩和保留重要信息。

使用示例:
```python
from app.agent.context import (
    count_tokens,
    truncate_messages,
    compress_context,
    ContextManager,
)

# 计算 Token 数
token_count = count_tokens("Hello, world!", model="gpt-4o")

# 截断消息列表
truncated = truncate_messages(messages, max_tokens=4000)

# 压缩上下文
compressed = await compress_context(messages, target_tokens=2000)

# 使用上下文管理器
manager = ContextManager(max_tokens=8000)
manager.add_messages(messages)
await manager.optimize()
```
"""

import asyncio
import logging
import re
from collections import OrderedDict
from datetime import datetime
from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.config.settings import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ============== Token 计算 ==============

# Token 估算常量
# 参考: OpenAI 的 tokenization 规则
_CHARS_PER_TOKEN = 4  # 英文平均每个 token 约 4 个字符
_CHINESE_CHARS_PER_TOKEN = 1.5  # 中文平均每个 token 约 1.5 个字符


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """计算文本的 Token 数量

    使用启发式算法估算 Token 数，无需调用 API。
    对于精确计算，可以使用 tiktoken 库。

    Args:
        text: 输入文本
        model: 模型名称（用于选择 Tokenization 策略）

    Returns:
        估算的 Token 数量

    Examples:
        >>> count_tokens("Hello, world!")
        4
        >>> count_tokens("你好世界", model="gpt-4o")
        3
    """
    if not text:
        return 0

    # 统计中文字符
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    # 统计英文字符（非中文）
    english_chars = len(text) - chinese_chars

    # 估算 Token 数
    chinese_tokens = int(chinese_chars / _CHINESE_CHARS_PER_TOKEN)
    english_tokens = int(english_chars / _CHARS_PER_TOKEN)

    total_tokens = chinese_tokens + english_tokens

    logger.debug(
        "tokens_counted",
        model=model,
        chinese_chars=chinese_chars,
        english_chars=english_chars,
        total_tokens=total_tokens,
    )

    return total_tokens


def count_messages_tokens(messages: list[BaseMessage], model: str = "gpt-4o") -> int:
    """计算消息列表的总 Token 数

    Args:
        messages: 消息列表
        model: 模型名称

    Returns:
        总 Token 数量
    """
    total = 0

    for message in messages:
        # 消息本身的内容
        content = message.content
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            # 多模态内容
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total += count_tokens(item["text"], model)

        # 消息元数据的 Token 开销（约 3-5 tokens per message）
        total += 4

        # 工具调用消息的开销
        if hasattr(message, "tool_calls") and message.tool_calls:
            total += len(message.tool_calls) * 10

        # 工具返回值的开销
        if isinstance(message, ToolMessage):
            total += 5

    logger.debug("messages_tokens_counted", message_count=len(messages), total_tokens=total)
    return total


# ============== 消息截断 ==============

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


# ============== 上下文压缩 ==============

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


# ============== 上下文管理器 ==============

class ContextManager:
    """上下文管理器

    管理对话上下文，自动优化 Token 使用。
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        model: str = "gpt-4o",
        reserve_ratio: float = 0.1,
    ):
        """初始化上下文管理器

        Args:
            max_tokens: 最大 Token 数
            model: 模型名称
            reserve_ratio: 预留比例（用于响应输出）
        """
        self.max_tokens = max_tokens
        self.model = model
        self.reserve_ratio = reserve_ratio
        self.effective_max = int(max_tokens * (1 - reserve_ratio))

        self.messages: OrderedDict[str, BaseMessage] = OrderedDict()
        self._message_counter = 0

    def add_message(self, message: BaseMessage) -> None:
        """添加消息

        Args:
            message: 消息
        """
        message_id = f"msg_{self._message_counter}"
        self.messages[message_id] = message
        self._message_counter += 1

        logger.debug(
            "message_added_to_context",
            message_id=message_id,
            message_type=type(message).__name__,
            total_messages=len(self.messages),
        )

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """批量添加消息

        Args:
            messages: 消息列表
        """
        for msg in messages:
            self.add_message(msg)

    async def optimize(self) -> list[BaseMessage]:
        """优化上下文（压缩/截断）

        Returns:
            优化后的消息列表

        Examples:
            >>> manager = ContextManager(max_tokens=4000)
            >>> manager.add_messages(long_conversation)
            >>> optimized = await manager.optimize()
        """
        message_list = list(self.messages.values())

        current_tokens = count_messages_tokens(message_list, self.model)

        if current_tokens <= self.effective_max:
            logger.debug("context_optimization_not_needed", current_tokens=current_tokens)
            return message_list

        logger.info(
            "context_optimization_needed",
            current_tokens=current_tokens,
            max_tokens=self.effective_max,
        )

        # 使用压缩器
        compressor = ContextCompressor(
            target_tokens=self.effective_max,
            model=self.model,
        )

        optimized = await compressor.compress(message_list)

        logger.info(
            "context_optimized",
            original_count=len(message_list),
            optimized_count=len(optimized),
            original_tokens=current_tokens,
            optimized_tokens=count_messages_tokens(optimized, self.model),
        )

        return optimized

    def get_messages(self) -> list[BaseMessage]:
        """获取当前所有消息

        Returns:
            消息列表
        """
        return list(self.messages.values())

    def clear(self) -> None:
        """清空上下文"""
        self.messages.clear()
        self._message_counter = 0
        logger.debug("context_cleared")

    def get_token_count(self) -> int:
        """获取当前 Token 数

        Returns:
            Token 数量
        """
        return count_messages_tokens(self.get_messages(), self.model)


# ============== 滑动窗口 ==============

class SlidingContextWindow:
    """滑动窗口上下文管理

    维护固定大小的上下文窗口，自动移除旧消息。
    """

    def __init__(
        self,
        window_size: int,
        max_tokens: int,
        model: str = "gpt-4o",
    ):
        """初始化滑动窗口

        Args:
            window_size: 窗口大小（消息数）
            max_tokens: 最大 Token 数
            model: 模型名称
        """
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.model = model
        self._messages: list[BaseMessage] = []

    def add(self, message: BaseMessage) -> None:
        """添加消息

        Args:
            message: 消息
        """
        self._messages.append(message)

        # 移除超出窗口大小的消息
        if len(self._messages) > self.window_size:
            removed = self._messages[:-self.window_size]
            self._messages = self._messages[-self.window_size:]

            logger.debug(
                "messages_evicted_from_window",
                evicted_count=len(removed),
                window_size=self.window_size,
            )

        # 检查 Token 限制
        while count_messages_tokens(self._messages, self.model) > self.max_tokens:
            if len(self._messages) <= 1:
                break

            removed = self._messages.pop(0)
            logger.debug(
                "message_evicted_due_to_token_limit",
                message_type=type(removed).__name__,
            )

    def get_messages(self) -> list[BaseMessage]:
        """获取当前窗口内的消息

        Returns:
            消息列表
        """
        return self._messages.copy()

    def is_full(self) -> bool:
        """检查窗口是否已满

        Returns:
            是否已满
        """
        return len(self._messages) >= self.window_size


# ============== Token 估算器（精确版）=============

# 尝试导入 tiktoken
try:
    import tiktoken

    _tiktoken_available = True
    _encoding_cache: dict[str, Any] = {}

    def _get_encoding(model: str) -> Any:
        """获取 TikToken 编码器

        Args:
            model: 模型名称

        Returns:
            TikToken Encoding 实例
        """
        if model in _encoding_cache:
            return _encoding_cache[model]

        # 映射模型名称到编码器
        encoding_map = {
            "gpt-4o": "o200k_base",
            "gpt-4o-mini": "o200k_base",
            "gpt-4": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "text-davinci-003": "p50k_base",
        }

        encoding_name = encoding_map.get(model, "cl100k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        _encoding_cache[model] = encoding

        return encoding

    def count_tokens_precise(text: str, model: str = "gpt-4o") -> int:
        """精确计算 Token 数（使用 tiktoken）

        Args:
            text: 输入文本
            model: 模型名称

        Returns:
            Token 数量
        """
        if not text:
            return 0

        try:
            encoding = _get_encoding(model)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning("tiktoken_failed", error=str(e), using_estimate=True)
            return count_tokens(text, model)

except ImportError:
    _tiktoken_available = False
    logger.warning("tiktoken_not_installed")

    def count_tokens_precise(text: str, model: str = "gpt-4o") -> int:
        """精确计算 Token 数（回退到估算）

        Args:
            text: 输入文本
            model: 模型名称

        Returns:
            Token 数量
        """
        return count_tokens(text, model)


__all__ = [
    # Token 计算
    "count_tokens",
    "count_messages_tokens",
    "count_tokens_precise",
    # 消息截断
    "truncate_messages",
    "truncate_text",
    # 上下文压缩
    "ContextCompressor",
    "compress_context",
    # 上下文管理器
    "ContextManager",
    "SlidingContextWindow",
    # 工具函数
    "_tiktoken_available",
    "_get_encoding",
]
