"""滑动窗口上下文管理模块

提供固定大小的滑动窗口上下文管理。

使用示例:
```python
from app.agent.context.sliding_window import SlidingContextWindow

# 创建滑动窗口
window = SlidingContextWindow(window_size=10, max_tokens=4000)
window.add(message)

# 获取窗口内消息
messages = window.get_messages()
```
"""

from langchain_core.messages import BaseMessage

from app.agent.context.token_counter import count_messages_tokens
from app.observability.logging import get_logger

logger = get_logger(__name__)


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


__all__ = [
    "SlidingContextWindow",
]
