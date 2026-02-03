"""ChatAgent 状态定义

整合了原 state.py 和 graph/types.py 中的 ChatState 定义。
"""

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState

from app.agent.context import count_messages_tokens
from app.observability.logging import get_logger

logger = get_logger(__name__)

# 默认上下文窗口配置
DEFAULT_MAX_TOKENS = 8000  # 大多数 LLM 的默认上下文窗口
DEFAULT_MAX_MESSAGES = 50  # 最大消息数量限制
TRUNCATION_STRATEGY = "sliding"  # 截断策略：sliding（滑动窗口）或 truncate（简单截断）


class ChatState(MessagesState):
    """聊天状态（扩展 MessagesState）

    继承 LangGraph 的 MessagesState，自动包含 messages 字段
    并使用 add_messages reducer 管理消息历史。

    新增功能：
    - 上下文窗口管理（自动截断超长历史）
    - Token 计数和限制
    - 滑动窗口保留最近消息

    Attributes:
        messages: 消息列表（继承自 MessagesState，自动使用 add_messages reducer）
        user_id: 用户 ID
        session_id: 会话 ID
        tenant_id: 租户 ID
        iteration_count: 迭代计数（用于防止无限循环）
        max_iterations: 最大迭代次数
        max_tokens: 最大 token 数（上下文窗口）
        max_messages: 最大消息数量
        error: 错误信息
    """

    # 用户和会话信息
    user_id: str | None = None
    session_id: str = ""
    tenant_id: int | None = None

    # 迭代控制
    iteration_count: int = 0
    max_iterations: int = 10

    # 上下文窗口管理
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_messages: int = DEFAULT_MAX_MESSAGES

    # 错误处理
    error: str | None = None

    def trim_messages(self) -> list[BaseMessage]:
        """根据配置截断消息列表

        使用滑动窗口策略，保留最近的重要消息。

        Returns:
            截断后的消息列表
        """
        messages = self.get("messages", [])

        if len(messages) > self.max_messages:
            logger.debug(
                "trimming_messages_by_count",
                current_count=len(messages),
                max_count=self.max_messages,
            )
            messages = messages[-self.max_messages :]

        token_count = count_messages_tokens(messages)
        if token_count > self.max_tokens:
            logger.debug(
                "trimming_messages_by_tokens",
                current_tokens=token_count,
                max_tokens=self.max_tokens,
            )
            messages = self._sliding_window_trim(messages, self.max_tokens)

        return messages

    @staticmethod
    def _sliding_window_trim(
        messages: list[BaseMessage],
        max_tokens: int,
    ) -> list[BaseMessage]:
        """使用滑动窗口截断消息

        从最旧的消息开始删除，直到 token 数满足限制。
        保留系统消息（如果有）和最近的对话。

        Args:
            messages: 消息列表
            max_tokens: 最大 token 数

        Returns:
            截断后的消息列表
        """
        from langchain_core.messages import SystemMessage

        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        regular_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        while regular_messages and count_messages_tokens(
            system_messages + regular_messages
        ) > max_tokens:
            removed = regular_messages.pop(0)
            logger.debug(
                "removed_message",
                type=getattr(removed, "type", "unknown"),
                content_length=len(getattr(removed, "content", "")),
            )

        return system_messages + regular_messages


def create_chat_state(
    messages: list[BaseMessage] | None = None,
    user_id: str | None = None,
    session_id: str = "",
    tenant_id: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_messages: int = DEFAULT_MAX_MESSAGES,
) -> ChatState:
    """创建聊天状态

    Args:
        messages: 初始消息列表
        user_id: 用户 ID
        session_id: 会话 ID
        tenant_id: 租户 ID
        max_tokens: 最大 token 数
        max_messages: 最大消息数量

    Returns:
        ChatState 实例
    """
    return ChatState(
        messages=messages or [],
        user_id=user_id,
        session_id=session_id,
        tenant_id=tenant_id,
        iteration_count=0,
        max_iterations=10,
        max_tokens=max_tokens,
        max_messages=max_messages,
        error=None,
    )


def create_state_from_input(
    input_data: str | dict[str, Any],
    session_id: str | None = None,
    user_id: str | None = None,
    tenant_id: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_messages: int = DEFAULT_MAX_MESSAGES,
) -> ChatState:
    """从输入创建状态

    兼容旧版 API，返回 ChatState。

    Args:
        input_data: 输入数据（字符串或字典）
        session_id: 会话 ID
        user_id: 用户 ID
        tenant_id: 租户 ID
        max_tokens: 最大 token 数
        max_messages: 最大消息数量

    Returns:
        ChatState 实例
    """
    if isinstance(input_data, str):
        query = input_data
    else:
        query = input_data.get("query", input_data.get("message", ""))

    from langchain_core.messages import HumanMessage

    messages = [HumanMessage(content=query)] if query else []

    return ChatState(
        messages=messages,
        user_id=user_id,
        session_id=session_id or "",
        tenant_id=tenant_id,
        iteration_count=0,
        max_iterations=10,
        max_tokens=max_tokens,
        max_messages=max_messages,
        error=None,
    )


__all__ = [
    "ChatState",
    "create_chat_state",
    "create_state_from_input",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_MESSAGES",
]
