"""记忆仓储

提供 LangGraph Store 持久化的异步数据访问层。
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Memory
from app.observability.logging import get_logger

logger = get_logger(__name__)


class MemoryRepository:
    """记忆仓储（异步版本）

    实现 LangGraph Store 接口，用于长期记忆存储。
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化仓储

        Args:
            session: 异步数据库会话
        """
        self.session = session

    async def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        """存储记忆

        Args:
            namespace: 命名空间
            key: 键
            value: 值（JSON）
        """
        try:
            statement = select(Memory).where(
                Memory.namespace == namespace,
                Memory.key == key,
            )
            result = await self.session.execute(statement)
            memory = result.scalar_one_or_none()

            if memory:
                memory.value = value
            else:
                memory = Memory(namespace=namespace, key=key, value=value)
                self.session.add(memory)

            await self.session.commit()
            logger.debug("memory_stored", namespace=namespace, key=key)

        except Exception as e:
            await self.session.rollback()
            logger.error("memory_put_failed", namespace=namespace, key=key, error=str(e))
            raise

    async def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        """获取记忆

        Args:
            namespace: 命名空间
            key: 键

        Returns:
            记忆值或 None
        """
        try:
            statement = select(Memory).where(
                Memory.namespace == namespace,
                Memory.key == key,
            )
            result = await self.session.execute(statement)
            memory = result.scalar_one_or_none()
            return memory.value if memory else None

        except Exception as e:
            logger.error("memory_get_failed", namespace=namespace, key=key, error=str(e))
            return None

    async def list(self, namespace: str) -> list[Memory]:
        """列出命名空间下的所有记忆

        Args:
            namespace: 命名空间

        Returns:
            记忆列表
        """
        try:
            statement = select(Memory).where(Memory.namespace == namespace)
            result = await self.session.execute(statement)
            return list(result.scalars().all())

        except Exception as e:
            logger.error("memory_list_failed", namespace=namespace, error=str(e))
            return []

    async def delete(self, namespace: str, key: str) -> bool:
        """删除记忆

        Args:
            namespace: 命名空间
            key: 键

        Returns:
            是否删除成功
        """
        try:
            statement = select(Memory).where(
                Memory.namespace == namespace,
                Memory.key == key,
            )
            result = await self.session.execute(statement)
            memory = result.scalar_one_or_none()

            if memory:
                await self.session.delete(memory)
                await self.session.commit()
                logger.debug("memory_deleted", namespace=namespace, key=key)
                return True

            return False

        except Exception as e:
            await self.session.rollback()
            logger.error("memory_delete_failed", namespace=namespace, key=key, error=str(e))
            return False

    async def clear_namespace(self, namespace: str) -> int:
        """清空命名空间下的所有记忆

        Args:
            namespace: 命名空间

        Returns:
            删除的数量
        """
        try:
            statement = select(Memory).where(Memory.namespace == namespace)
            result = await self.session.execute(statement)
            memories = result.scalars().all()

            count = 0
            for memory in memories:
                await self.session.delete(memory)
                count += 1

            await self.session.commit()
            logger.info("memory_namespace_cleared", namespace=namespace, count=count)
            return count

        except Exception as e:
            await self.session.rollback()
            logger.error("memory_clear_namespace_failed", namespace=namespace, error=str(e))
            return 0

    async def delete_expired(self) -> int:
        """删除过期的记忆

        Returns:
            删除的数量
        """
        try:
            from datetime import UTC, datetime

            statement = select(Memory).where(
                Memory.expires_at.is_not(None),
                Memory.expires_at < datetime.now(UTC),  # type: ignore
            )
            result = await self.session.execute(statement)
            memories = result.scalars().all()

            count = 0
            for memory in memories:
                await self.session.delete(memory)
                count += 1

            await self.session.commit()
            logger.info("memory_expired_deleted", count=count)
            return count

        except Exception as e:
            await self.session.rollback()
            logger.error("memory_delete_expired_failed", error=str(e))
            return 0


class StoreAdapter:
    """LangGraph Store 适配器

    将 MemoryRepository 适配为 LangGraph Store 接口。
    LangGraph Store 用于存储和检索跨会话的记忆。
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化适配器

        Args:
            session: 异步数据库会话
        """
        self._memory_repo = MemoryRepository(session)

    async def aget(self, namespace: str, key: str) -> Any | None:
        """异步获取记忆"""
        return await self._memory_repo.get(namespace, key)

    async def aput(
        self,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """异步存储记忆"""
        await self._memory_repo.put(namespace, key, value)

    async def adelete(self, namespace: str, key: str) -> None:
        """异步删除记忆"""
        await self._memory_repo.delete(namespace, key)

    async def alist(self, namespace: str) -> list[tuple[str, Any]]:
        """异步列出命名空间下的所有记忆"""
        memories = await self._memory_repo.list(namespace)
        return [(m.key, m.value) for m in memories]
