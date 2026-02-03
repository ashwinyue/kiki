"""Agent 基类

定义所有 Agent 的统一接口，确保不同 Agent 类型有一致的 API。

设计原则:
- 单一职责: 只定义核心接口
- 开闭原则: 易于扩展新的 Agent 类型
- 依赖倒置: 依赖抽象而非具体实现
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from langchain_core.messages import BaseMessage


class BaseAgent(ABC):
    """Agent 抽象基类

    定义所有 Agent 必须实现的核心接口。

    核心方法:
        - get_response(): 获取完整响应
        - astream(): 流式响应

    生命周期管理:
        - 支持异步上下文管理器
        - 自动资源清理

    示例:
        ```python
        # 异步上下文管理器模式（推荐）
        async with ChatAgent(system_prompt="...") as agent:
            response = await agent.get_response("你好", session_id="session-123")

        # 手动关闭模式
        agent = ChatAgent(system_prompt="...")
        try:
            response = await agent.get_response("你好", session_id="session-123")
        finally:
            await agent.close()
        ```
    """

    @abstractmethod
    async def get_response(
        self,
        message: str,
        session_id: str,
        user_id: str | None = None,
        tenant_id: int | None = None,
    ) -> list[BaseMessage]:
        """获取 Agent 响应

        Args:
            message: 用户消息
            session_id: 会话 ID（用于状态持久化）
            user_id: 用户 ID（可选）
            tenant_id: 租户 ID（可选）

        Returns:
            消息列表（包含 AI 响应）

        Raises:
            RuntimeError: Agent 未初始化或调用失败

        Examples:
            ```python
            messages = await agent.get_response(
                message="今天天气怎么样？",
                session_id="session-123",
            )
            ai_message = messages[-1]  # 最后一条是 AI 响应
            print(ai_message.content)
            ```
        """
        pass

    @abstractmethod
    async def astream(
        self,
        message: str,
        session_id: str,
        user_id: str | None = None,
        tenant_id: int | None = None,
    ) -> AsyncIterator[BaseMessage]:
        """流式获取 Agent 响应

        Args:
            message: 用户消息
            session_id: 会话 ID
            user_id: 用户 ID（可选）
            tenant_id: 租户 ID（可选）

        Yields:
            消息（逐个产出）

        Examples:
            ```python
            async for msg in agent.astream("你好", session_id="session-123"):
                if msg.type == "ai":
                    print(msg.content, end="", flush=True)
            ```
        """
        pass

    # 生命周期管理

    async def __aenter__(self) -> BaseAgent:
        """异步上下文管理器入口

        Returns:
            self
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口

        确保资源正确释放。
        """
        await self.close()

    async def close(self) -> None:
        """关闭 Agent，释放资源

        默认实现为空，子类可覆盖以释放资源（如连接池）。

        Examples:
            ```python
            class MyAgent(BaseAgent):
                async def close(self) -> None:
                    # 关闭连接池
                    if self._connection_pool:
                        await self._connection_pool.close()
            ```
        """
        pass

    # 可选方法（子类可覆盖）

    async def get_session_history(
        self,
        session_id: str,
    ) -> list[BaseMessage]:
        """获取会话历史

        Args:
            session_id: 会话 ID

        Returns:
            消息历史

        Note:
            默认返回空列表，子类可覆盖以从数据库加载历史。
        """
        return []

    async def clear_session(
        self,
        session_id: str,
    ) -> None:
        """清除会话历史

        Args:
            session_id: 会话 ID

        Note:
            默认不做任何操作，子类可覆盖以清除数据库中的历史。
        """
        pass


__all__ = ["BaseAgent"]
