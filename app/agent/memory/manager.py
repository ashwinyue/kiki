"""Memory 管理器

统一管理短期和长期记忆，使用依赖注入模式。
"""

from typing import Any

from langchain_core.messages import BaseMessage

from app.agent.memory.base import BaseLongTermMemory
from app.agent.memory.short_term import ShortTermMemory
from app.observability.logging import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Memory 管理器

    提供统一的记忆管理接口，协调短期和长期记忆。
    使用依赖注入模式，而非延迟导入。
    """

    def __init__(
        self,
        session_id: str,
        user_id: str | None = None,
        long_term_memory: BaseLongTermMemory | None = None,
    ) -> None:
        """初始化 Memory Manager

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            long_term_memory: 长期记忆实例（可选，通过依赖注入）
        """
        self.session_id = session_id
        self.user_id = user_id
        self._long_term = long_term_memory

        # 短期记忆（始终创建）
        self.short_term = ShortTermMemory(session_id)

        logger.info(
            "memory_manager_initialized",
            session_id=session_id,
            user_id=user_id,
            has_long_term=long_term_memory is not None,
        )

    @property
    def has_long_term(self) -> bool:
        """是否启用长期记忆"""
        return self._long_term is not None

    async def add_message(self, message: BaseMessage) -> None:
        """添加消息到短期记忆

        Args:
            message: 消息对象
        """
        await self.short_term.add_message(message)
        logger.debug("message_added", type=message.type)

    async def add_long_term_memory(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """添加长期记忆

        Args:
            content: 记忆内容
            metadata: 元数据

        Returns:
            记忆 ID，如果未启用长期记忆则返回 None
        """
        if self._long_term is None:
            logger.debug("long_term_memory_not_available")
            return None

        metadata = metadata or {}
        metadata.update(
            {
                "session_id": self.session_id,
                "user_id": self.user_id,
            }
        )

        memory_id = await self._long_term.add_memory(content, metadata)
        logger.info("long_term_memory_added", memory_id=memory_id)
        return memory_id

    async def search_long_term(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """搜索长期记忆

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            记忆列表
        """
        if self._long_term is None:
            return []

        # 添加用户/会话过滤
        filter = filter or {}
        filter.update(
            {
                "session_id": self.session_id,
                "user_id": self.user_id,
            }
        )

        results = await self._long_term.search_memories(query, k=k, filter=filter)
        logger.debug("long_term_memory_searched", result_count=len(results))
        return results

    async def get_short_term_messages(self) -> list[BaseMessage]:
        """获取短期消息

        Returns:
            消息列表
        """
        return await self.short_term.get_messages()

    async def clear_short_term(self) -> None:
        """清除短期记忆"""
        await self.short_term.clear()
        logger.info("short_term_memory_cleared", session_id=self.session_id)

    async def close(self) -> None:
        """关闭 Memory Manager，释放资源"""
        await self.short_term.close()

        if self._long_term is not None:
            if hasattr(self._long_term, "close"):
                await self._long_term.close()

        logger.debug("memory_manager_closed", session_id=self.session_id)


def create_memory_manager(
    session_id: str,
    user_id: str | None = None,
    long_term_memory: BaseLongTermMemory | None = None,
) -> MemoryManager:
    """创建 Memory Manager 实例

    Args:
        session_id: 会话 ID
        user_id: 用户 ID
        long_term_memory: 长期记忆实例（可选，通过依赖注入）

    Returns:
        MemoryManager 实例
    """
    return MemoryManager(
        session_id=session_id,
        user_id=user_id,
        long_term_memory=long_term_memory,
    )


class MemoryManagerFactory:
    """Memory Manager 工厂

    集中管理 Memory Manager 的创建和配置。
    """

    _default_long_term_memory: BaseLongTermMemory | None = None

    @classmethod
    def set_default_long_term_memory(cls, long_term: BaseLongTermMemory) -> None:
        """设置默认长期记忆实例

        Args:
            long_term: 长期记忆实例
        """
        cls._default_long_term_memory = long_term
        logger.info("default_long_term_memory_set")

    @classmethod
    def create(
        cls,
        session_id: str,
        user_id: str | None = None,
        use_long_term: bool = False,
    ) -> MemoryManager:
        """创建 Memory Manager

        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            use_long_term: 是否使用长期记忆

        Returns:
            MemoryManager 实例
        """
        long_term = cls._default_long_term_memory if use_long_term else None
        return MemoryManager(session_id, user_id, long_term)
