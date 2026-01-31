"""线程仓储

提供线程相关的数据访问操作。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import Thread
from app.observability.logging import get_logger
from app.repositories.base import BaseRepository, PaginatedResult, PaginationParams

logger = get_logger(__name__)


class ThreadRepository(BaseRepository[Thread]):
    """线程仓储类

    线程用于 LangGraph 状态持久化。
    """

    def __init__(self, session: AsyncSession):
        """初始化线程仓储

        Args:
            session: 异步数据库会话
        """
        super().__init__(Thread, session)

    async def get_by_id(
        self,
        thread_id: str,
    ) -> Thread | None:
        """根据线程 ID 获取线程

        Args:
            thread_id: 线程 ID

        Returns:
            线程实例或 None
        """
        return await self.get(thread_id)

    async def create_thread(
        self,
        thread_id: str,
        name: str,
        user_id: int | None = None,
    ) -> Thread:
        """创建线程

        Args:
            thread_id: 线程 ID
            name: 线程名称
            user_id: 用户 ID

        Returns:
            创建的线程实例
        """
        thread = Thread(
            id=thread_id,
            name=name,
            user_id=user_id,
            status="active",
        )
        return await self.create(thread)

    async def list_by_user(
        self,
        user_id: int,
        params: PaginationParams,
    ) -> PaginatedResult[Thread]:
        """分页获取用户的线程列表

        Args:
            user_id: 用户 ID
            params: 分页参数

        Returns:
            分页结果
        """
        return await self.list_paginated(params, user_id=user_id)

    async def list_by_status(
        self,
        status: str,
        params: PaginationParams,
    ) -> PaginatedResult[Thread]:
        """分页获取指定状态的线程列表

        Args:
            status: 状态 (active/archived/deleted)
            params: 分页参数

        Returns:
            分页结果
        """
        return await self.list_paginated(params, status=status)

    async def update_status(
        self,
        thread_id: str,
        status: str,
    ) -> bool:
        """更新线程状态

        Args:
            thread_id: 线程 ID
            status: 新状态

        Returns:
            是否更新成功
        """
        thread = await self.get(thread_id)
        if thread is None:
            return False
        thread.status = status
        await self.session.commit()
        logger.info("thread_status_updated", thread_id=thread_id, status=status)
        return True

    async def archive(
        self,
        thread_id: str,
    ) -> bool:
        """归档线程

        Args:
            thread_id: 线程 ID

        Returns:
            是否归档成功
        """
        return await self.update_status(thread_id, "archived")

    async def delete_soft(
        self,
        thread_id: str,
    ) -> bool:
        """软删除线程

        Args:
            thread_id: 线程 ID

        Returns:
            是否删除成功
        """
        return await self.update_status(thread_id, "deleted")
