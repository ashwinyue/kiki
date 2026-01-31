"""窗口记忆（Window Memory）模块

基于 LangChain 的 trim_messages 实现 Token 级别的滑动窗口记忆机制。

核心特性：
- Token 级别限制（而非消息数量）
- 支持多种修剪策略（last/first）
- 确保对话边界完整性
- 支持自定义 token 计数器

参考实现: week07/p07-windowMEM.py
"""

from enum import Enum
from typing import Callable, Literal

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)

from app.agent.state import AgentState
from app.core.config import get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


class TrimStrategy(str, Enum):
    """修剪策略枚举"""

    LAST = "last"  # 保留最后（最新）的消息
    FIRST = "first"  # 保留最开始的消息


class TokenCounterType(str, Enum):
    """Token 计数器类型"""

    APPROXIMATE = "approximate"  # 近似计数（快速）
    EXACT = "exact"  # 精确计数（需要 tiktoken）


def _get_token_counter(
    counter_type: TokenCounterType = TokenCounterType.APPROXIMATE,
) -> Callable[[BaseMessage], int]:
    """获取 Token 计数器函数

    Args:
        counter_type: 计数器类型

    Returns:
        Token 计数函数
    """
    if counter_type == TokenCounterType.EXACT:
        try:
            from langchain_core.messages.utils import count_tokens

            return count_tokens
        except ImportError:
            logger.warning(
                "exact_token_counter_not_available",
                fallback="approximate",
            )
            return count_tokens_approximately

    return count_tokens_approximately


def create_pre_model_hook(
    max_tokens: int | None = None,
    strategy: Literal["last", "first"] = "last",
    token_counter: Callable[[BaseMessage], int] | None = None,
    start_on: str | tuple[str, ...] | None = None,
    end_on: str | tuple[str, ...] | None = None,
    preserve_system: bool = True,
) -> Callable[[AgentState], dict]:
    """创建 pre_model_hook 钩子函数

    这个钩子会在 LLM 调用之前被调用，用于修剪消息历史，
    确保 Token 数量不超过限制。

    Args:
        max_tokens: 最大 Token 数量（None 则从配置读取）
        strategy: 修剪策略 ("last" 保留最新，"first" 保留最旧)
        token_counter: Token 计数函数（None 则使用近似计数）
        start_on: 从哪种类型的消息开始计数
        end_on: 在哪种类型的消息结束计数
        preserve_system: 是否保留系统消息

    Returns:
        pre_model_hook 钩子函数

    Examples:
        ```python
        # 简单使用
        hook = create_pre_model_hook(max_tokens=384)

        # 自定义策略
        hook = create_pre_model_hook(
            max_tokens=1000,
            strategy="last",
            start_on="human",
            end_on=("human", "tool"),
        )

        # 使用于 LangGraph
        builder = StateGraph(AgentState)
        builder.add_node(
            "agent",
            model_node,
            pre_model_hook=hook,  # 添加钩子
        )
        ```
    """
    settings = get_settings()
    max_tokens = max_tokens or settings.context_max_tokens

    if token_counter is None:
        token_counter = _get_token_counter(TokenCounterType.APPROXIMATE)

    # 默认值：从人类消息开始，在人类或工具消息结束
    if start_on is None:
        start_on = "human"
    if end_on is None:
        end_on = ("human", "tool")

    def pre_model_hook(state: AgentState) -> dict:
        """窗口记忆预处理钩子

        在 LLM 调用前修剪消息，确保 Token 数量不超过限制。

        Args:
            state: 当前 Agent 状态

        Returns:
            包含修剪后消息的状态更新
        """
        messages = state.get("messages", [])

        if not messages:
            return {"llm_input_messages": []}

        # 检查是否需要修剪
        # 先用近似计数快速判断
        total_tokens = sum(token_counter(msg) for msg in messages)

        if total_tokens <= max_tokens:
            logger.debug(
                "window_memory_no_trim_needed",
                token_count=total_tokens,
                max_tokens=max_tokens,
            )
            return {"llm_input_messages": messages}

        # 需要修剪
        logger.info(
            "window_memory_trim_triggered",
            token_count=total_tokens,
            max_tokens=max_tokens,
            message_count_before=len(messages),
            strategy=strategy,
        )

        try:
            trimmed_messages = trim_messages(
                messages,
                strategy=strategy,
                token_counter=token_counter,
                max_tokens=max_tokens,
                start_on=start_on,
                end_on=end_on,
            )

            # 处理系统消息保留
            if preserve_system:
                system_messages = [m for m in messages if m.type == "system"]
                other_trimmed = [m for m in trimmed_messages if m.type != "system"]

                # 如果修剪后的结果没有系统消息，但原消息有，则添加回去
                if system_messages and not any(m.type == "system" for m in trimmed_messages):
                    # 只添加第一条系统消息
                    trimmed_messages = [system_messages[0]] + other_trimmed

            logger.info(
                "window_memory_trim_complete",
                message_count_after=len(trimmed_messages),
                tokens_removed=total_tokens - sum(token_counter(m) for m in trimmed_messages),
            )

            return {"llm_input_messages": trimmed_messages}

        except Exception as e:
            logger.exception(
                "window_memory_trim_failed",
                error=str(e),
            )
            # 失败时返回原消息，避免中断流程
            return {"llm_input_messages": messages}

    return pre_model_hook


class WindowMemoryManager:
    """窗口记忆管理器

    提供更高级的窗口记忆管理功能：
    - 动态调整窗口大小
    - 统计 Token 使用情况
    - 支持多种修剪策略

    Examples:
        ```python
        manager = WindowMemoryManager(max_tokens=1000)

        # 修剪消息
        trimmed = await manager.trim_messages(messages)

        # 获取当前 Token 统计
        stats = manager.get_stats()
        ```
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        strategy: Literal["last", "first"] = "last",
        token_counter_type: TokenCounterType = TokenCounterType.APPROXIMATE,
        preserve_system: bool = True,
    ) -> None:
        """初始化窗口记忆管理器

        Args:
            max_tokens: 最大 Token 数量
            strategy: 修剪策略
            token_counter_type: Token 计数器类型
            preserve_system: 是否保留系统消息
        """
        settings = get_settings()
        self.max_tokens = max_tokens or settings.context_max_tokens
        self.strategy = strategy
        self.token_counter = _get_token_counter(token_counter_type)
        self.preserve_system = preserve_system

        # 统计信息
        self._total_trims = 0
        self._total_tokens_removed = 0

    def trim_messages(
        self,
        messages: list[BaseMessage],
        start_on: str | tuple[str, ...] | None = None,
        end_on: str | tuple[str, ...] | None = None,
    ) -> list[BaseMessage]:
        """修剪消息列表

        Args:
            messages: 原始消息列表
            start_on: 从哪种类型的消息开始计数
            end_on: 在哪种类型的消息结束计数

        Returns:
            修剪后的消息列表
        """
        if not messages:
            return []

        total_tokens = sum(self.token_counter(msg) for msg in messages)

        if total_tokens <= self.max_tokens:
            return messages

        try:
            trimmed = trim_messages(
                messages,
                strategy=self.strategy,
                token_counter=self.token_counter,
                max_tokens=self.max_tokens,
                start_on=start_on or "human",
                end_on=end_on or ("human", "tool"),
            )

            # 处理系统消息
            if self.preserve_system:
                system_messages = [m for m in messages if m.type == "system"]
                other_trimmed = [m for m in trimmed if m.type != "system"]

                if system_messages and not any(m.type == "system" for m in trimmed):
                    trimmed = [system_messages[0]] + other_trimmed

            # 更新统计
            self._total_trims += 1
            self._total_tokens_removed += total_tokens - sum(
                self.token_counter(m) for m in trimmed
            )

            logger.debug(
                "window_manager_trim",
                before_count=len(messages),
                after_count=len(trimmed),
                tokens_removed=total_tokens - sum(self.token_counter(m) for m in trimmed),
            )

            return trimmed

        except Exception as e:
            logger.exception("window_manager_trim_failed", error=str(e))
            return messages

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        """计算消息列表的总 Token 数

        Args:
            messages: 消息列表

        Returns:
            总 Token 数
        """
        return sum(self.token_counter(msg) for msg in messages)

    def get_stats(self) -> dict:
        """获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "max_tokens": self.max_tokens,
            "strategy": self.strategy,
            "preserve_system": self.preserve_system,
            "total_trims": self._total_trims,
            "total_tokens_removed": self._total_tokens_removed,
            "avg_tokens_removed": (
                self._total_tokens_removed / self._total_trims
                if self._total_trims > 0
                else 0
            ),
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._total_trims = 0
        self._total_tokens_removed = 0


# 全局单例管理器
_global_manager: WindowMemoryManager | None = None


def get_window_memory_manager() -> WindowMemoryManager:
    """获取全局窗口记忆管理器

    Returns:
        WindowMemoryManager 实例
    """
    global _global_manager
    if _global_manager is None:
        settings = get_settings()
        _global_manager = WindowMemoryManager(
            max_tokens=settings.context_max_tokens,
            strategy="last",
        )
    return _global_manager


# 便捷函数
def create_chat_hook(max_tokens: int | None = None) -> Callable[[AgentState], dict]:
    """创建聊天专用的 pre_model_hook

    这是 create_pre_model_hook 的简化版本，
    预设了适合聊天场景的参数。

    Args:
        max_tokens: 最大 Token 数

    Returns:
        pre_model_hook 函数
    """
    return create_pre_model_hook(
        max_tokens=max_tokens,
        strategy="last",
        start_on="human",
        end_on=("human", "tool"),
        preserve_system=True,
    )


def trim_state_messages(
    state: AgentState,
    max_tokens: int | None = None,
) -> list[BaseMessage]:
    """修剪状态中的消息

    便捷函数，用于直接修剪状态中的消息。

    Args:
        state: Agent 状态
        max_tokens: 最大 Token 数

    Returns:
        修剪后的消息列表
    """
    manager = get_window_memory_manager()
    if max_tokens:
        manager.max_tokens = max_tokens
    return manager.trim_messages(state.get("messages", []))


__all__ = [
    "TrimStrategy",
    "TokenCounterType",
    "create_pre_model_hook",
    "WindowMemoryManager",
    "get_window_memory_manager",
    "create_chat_hook",
    "trim_state_messages",
]
