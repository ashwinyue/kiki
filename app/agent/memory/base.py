"""Memory 抽象基类

定义所有 Memory 实现必须遵循的接口。
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage


class BaseMemory(ABC):
    """Memory 抽象基类

    定义短期和长期记忆的统一接口。
    """

    @abstractmethod
    async def add_message(self, message: BaseMessage) -> None:
        """添加消息到记忆

        Args:
            message: 消息对象
        """
        pass

    @abstractmethod
    async def get_messages(self, limit: int | None = None) -> list[BaseMessage]:
        """获取消息列表

        Args:
            limit: 最大返回数量

        Returns:
            消息列表
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """清除所有记忆"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """获取消息数量

        Returns:
            消息数量
        """
        pass


class BaseLongTermMemory(ABC):
    """长期记忆抽象基类

    支持语义检索的长期记忆接口。
    """

    @abstractmethod
    async def add_memory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加长期记忆

        Args:
            content: 记忆内容
            metadata: 元数据

        Returns:
            记忆 ID
        """
        pass

    @abstractmethod
    async def search_memories(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """语义检索记忆

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            记忆列表，每项包含 content、metadata、score 等
        """
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """删除记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            是否成功删除
        """
        pass

    @abstractmethod
    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆

        Args:
            memory_id: 记忆 ID
            content: 新内容（可选）
            metadata: 新元数据（可选）

        Returns:
            是否成功更新
        """
        pass
