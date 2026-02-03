"""Token 计算模块

提供 Token 计算功能，支持启发式估算和精确计算（使用 tiktoken）。

使用示例:
```python
from app.agent.context.token_counter import count_tokens, count_messages_tokens

# 计算文本 Token 数
token_count = count_tokens("Hello, world!", model="gpt-4o")

# 计算消息列表 Token 数
total = count_messages_tokens(messages, model="gpt-4o")
```
"""

import re
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from app.observability.logging import get_logger

logger = get_logger(__name__)

# ============== Token 估算常量 ==============

# Token 估算常量
# 参考: OpenAI 的 tokenization 规则
_CHARS_PER_TOKEN = 4  # 英文平均每个 token 约 4 个字符
_CHINESE_CHARS_PER_TOKEN = 1.5  # 中文平均每个 token 约 1.5 个字符


# ============== 启发式 Token 计算 ==============


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


# ============== TikToken 精确计算 ==============

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

    def _get_encoding(model: str) -> Any:  # type: ignore[misc]
        """获取 TikToken 编码器（未安装）

        Args:
            model: 模型名称

        Returns:
            None
        """
        return None

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
    # TikToken
    "_tiktoken_available",
    "_get_encoding",
]
