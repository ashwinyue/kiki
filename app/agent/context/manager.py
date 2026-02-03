"""上下文管理器模块

提供对话上下文的管理功能，自动优化 Token 使用。

使用示例:
```python
from app.agent.context.manager import ContextManager

# 创建上下文管理器
manager = ContextManager(max_tokens=8000)
manager.add_messages(messages)

# 优化上下文
optimized = await manager.optimize()
```
"""

from collections import OrderedDict

from langchain_core.messages import BaseMessage

from app.agent.context.compressor import ContextCompressor
from app.agent.context.token_counter import count_messages_tokens
from app.observability.logging import get_logger

logger = get_logger(__name__)


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


__all__ = [
    "ContextManager",
]
